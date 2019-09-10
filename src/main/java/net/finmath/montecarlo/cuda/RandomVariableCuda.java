/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 09.02.2006
 */
package net.finmath.montecarlo.cuda;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.IOException;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;

import jcuda.LogLevel;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import net.finmath.functions.DoubleTernaryOperator;
import net.finmath.montecarlo.RandomVariableFromDoubleArray;
import net.finmath.montecarlo.RandomVariableFromFloatArray;
import net.finmath.stochastic.RandomVariable;

/**
 * The class RandomVariableCuda represents a random variable being the evaluation of a stochastic process
 * at a certain time within a Monte-Carlo simulation.
 *
 * It is thus essentially a vector of floating point numbers - the realizations - together with a double - the time.
 * The index of the vector represents path.
 *
 * The class may also be used for non-stochastic quantities which may potentially be stochastic
 * (e.g. volatility). If only non-stochastic random variables are involved in an operation the class uses
 * optimized code.
 *
 * Accesses performed exclusively through the interface
 * <code>RandomVariable</code> is thread safe (and does not mutate the class).
 *
 * <b>This implementation uses floats for the realizations on a Cuda GPU.</b> There is a CPU implementation in {@link RandomVariableFromFloatArray} which give exactly the same results for all methods (checked by unit test).
 *
 * @author Christian Fries
 * @version 2.1
 */
public class RandomVariableCuda implements RandomVariable {

	/**
	 * An object referencing a cuda device pointer.
	 *
	 * This wrapper is mainly used to track, when references to device pointers are de-referenced, which then triggers
	 * a recycling of the device vector.
	 */
	public static class DevicePointerReference {
		private final CUdeviceptr devicePointer;

		public DevicePointerReference(final CUdeviceptr devicePointer) {
			this.devicePointer = devicePointer;
		}

		public CUdeviceptr get() {
			return devicePointer;
		}
	}

	/**
	 * A memory pool for the GPU vectors.
	 *
	 * The memory pool is provided for vectors of different length.
	 *
	 * Implementation details:
	 * The map vectorsToRecycleReferenceQueueMap maps each vector length to a ReferenceQueue<RandomVariableCuda> holding
	 * reference of recycleable vectors. The map vectorsInUseReferenceMap maps this weak reference to a Cuda vector.
	 *
	 * @author Christian Fries
	 */
	private static class DeviceMemoryPool {

		private final Object lock = new Object();

		/**
		 * For each vector size this map stores <code>ReferenceQueue&lt;DevicePointerReference&gt;</code>. The garbadge collector
		 * will put <code>WeakReference&lt;DevicePointerReference&gt;</code> into this queue once the
		 * <code>DevicePointerReference</code>-object has become de-referenced.
		 */
		private static final Map<Integer, ReferenceQueue<DevicePointerReference>>		vectorsToRecycleReferenceQueueMap	= new ConcurrentHashMap<Integer, ReferenceQueue<DevicePointerReference>>();

		/**
		 * This map allow to recover the device pointer for a given <code>WeakReference&lt;DevicePointerReference&gt;</code>.
		 */
		private static final Map<WeakReference<DevicePointerReference>, CUdeviceptr>	vectorsInUseReferenceMap			= new ConcurrentHashMap<WeakReference<DevicePointerReference>, CUdeviceptr>();

		/**
		 * Percentage of device memory at which we will trigger System.gc() to aggressively reduce references.
		 */
		private static final float	vectorsRecyclerPercentageFreeToStartGC		= 0.15f;		// should be set by monitoring GPU mem

		/**
		 * Percentage of device memory at which we will try to wait a few milliseconds for recycled objects.
		 */
		private static final float	vectorsRecyclerPercentageFreeToWaitForGC	= 0.05f;		// should be set by monitoring GPU mem

		/**
		 * Maximum time to wait for object recycled objects. (Higher value slows down the code, but prevents out-of-memory).
		 */
		private static final long	vectorsRecyclerMaxTimeOutMillis			= 1000;

		private static long	deviceAllocMemoryBytes = 0;
		private static long	deviceMaxMemoryBytes;

		/**
		 * Get a Java object ({@link DevicePointerReference}) representing a vector allocated on the GPU memory (device memory).
		 *
		 * If this object is the wrapped into a {@link RandomVariableCuda} via {@link RandomVariableCuda#of(double, DevicePointerReference, long)}
		 * you may perform arithmetic operations on it.
		 *
		 * Note: You will likely not use this method directly. Instead use {@link #getDevicePointer(float[])} which will
		 * call this method and initialize the vector to the given values.
		 *
		 * The object is "managed" in the sense the once the object is dereferenced the GPU memory will be marked for re-use (or freed at a later time).
		 *
		 * @param size The size of the vector as multiples of sizeof(float). (To allocated a double vector use twice the size).
		 * @return An object representing a vector allocated on the GPU memory.
		 */
		public DevicePointerReference getDevicePointer(final long size) {
			if(logger.isLoggable(Level.FINEST)) {
				final StringBuilder stringBuilder = new StringBuilder();
				stringBuilder.append("Memory pool stats: ");
				stringBuilder.append("  vector sizes: ");
				for(final Map.Entry<Integer, ReferenceQueue<DevicePointerReference>> entry : vectorsToRecycleReferenceQueueMap.entrySet()) {
					stringBuilder.append("    " + entry.getKey());
				}
				stringBuilder.append("  total number of vectors: " + vectorsInUseReferenceMap.size());
				logger.finest(stringBuilder.toString());
			}

			CUdeviceptr cuDevicePtr = null;

			// Check for object to recycle
			final ReferenceQueue<DevicePointerReference> vectorsToRecycleReferenceQueue = vectorsToRecycleReferenceQueueMap.computeIfAbsent(new Integer((int)size), key ->  {
				logger.info("Creating reference queue for vector size " + size);
				return new ReferenceQueue<DevicePointerReference>();
			});

			Reference<? extends DevicePointerReference> reference = vectorsToRecycleReferenceQueue.poll();
			if(reference != null) {
				cuDevicePtr = vectorsInUseReferenceMap.remove(reference);
				if(logger.isLoggable(Level.FINEST)) {
					logger.finest("Recycling (1) device pointer " + cuDevicePtr + " from " + reference);
				}
			}
			else {
				final float deviceFreeMemPercentage = getDeviceFreeMemPercentage();

				// No pointer found, try GC if we are above a critical level
				if(deviceFreeMemPercentage < vectorsRecyclerPercentageFreeToStartGC && deviceFreeMemPercentage >= vectorsRecyclerPercentageFreeToWaitForGC) {
					System.gc();
					System.runFinalization();

					if(logger.isLoggable(Level.FINEST)) {
						logger.fine("Device free memory " + deviceFreeMemPercentage*100 + "%");
					}

					reference = vectorsToRecycleReferenceQueue.poll();
				}

				// Wait for GC
				if(reference == null && deviceFreeMemPercentage < vectorsRecyclerPercentageFreeToWaitForGC) {
					/*
					 * Try to obtain a reference after GC, retry with waits for 1 ms, 10 ms, 100 ms, ...
					 */
					System.gc();

					long timeOut = 1;
					while(reference == null && timeOut < vectorsRecyclerMaxTimeOutMillis) {
						try {
							reference = vectorsToRecycleReferenceQueue.remove(timeOut);
							timeOut *= 4;
						} catch (IllegalArgumentException | InterruptedException e) {}
					}

					if(reference == null) {
						// Still no pointer found for requested size, consider cleaning all (also other sizes)
						logger.info("Last resort: Cleaning all unused vectors on device. Device free memory " + deviceFreeMemPercentage*100 + "%");
						clean();
					}
				}

				if(reference != null) {
					if(logger.isLoggable(Level.FINEST)) {
						logger.finest("Recycling (2) device pointer " + cuDevicePtr + " from " + reference);
					}
					cuDevicePtr = vectorsInUseReferenceMap.remove(reference);
				}
			}

			if(cuDevicePtr == null)  {
				// Still no pointer found, create new one
				try {
					cuDevicePtr =
							deviceExecutor.submit(new Callable<CUdeviceptr>() { @Override
								public CUdeviceptr call() {
								CUdeviceptr cuDevicePtr = new CUdeviceptr();
								final int succ = JCudaDriver.cuMemAlloc(cuDevicePtr, size * Sizeof.FLOAT);
								if(succ != 0) {
									cuDevicePtr = null;
									final String[] cudaErrorName = new String[1];
									JCudaDriver.cuGetErrorName(succ, cudaErrorName);
									final String[] cudaErrorDescription = new String[1];
									JCudaDriver.cuGetErrorString(succ, cudaErrorDescription);

									logger.warning("Failed creating device vector "+ cuDevicePtr + " with size=" + size + " with error "+ cudaErrorName + ": " + cudaErrorDescription);
								}
								return cuDevicePtr;
							}}).get();
				} catch (InterruptedException | ExecutionException e) {
					logger.severe("Failed to allocate device vector with size=" + size + ". Cause: " + e.getCause());
				}

				if(cuDevicePtr == null) {
					logger.severe("Failed to allocate device vector with size=" + size);
					throw new OutOfMemoryError("Failed to allocate device vector with size=" + size);
				}

				deviceAllocMemoryBytes += size * Sizeof.FLOAT;
			}

			/*
			 * Manage the pointer
			 */
			final DevicePointerReference devicePointerReference = new DevicePointerReference(cuDevicePtr);
			vectorsInUseReferenceMap.put(new WeakReference<DevicePointerReference>(devicePointerReference, vectorsToRecycleReferenceQueue), cuDevicePtr);

			return devicePointerReference;
		}

		/**
		 * Free all unused device memory.
		 */
		public void clean() {
			synchronized (lock) {
				// Clean up all remaining pointers
				for(final Entry<Integer, ReferenceQueue<DevicePointerReference>> entry : vectorsToRecycleReferenceQueueMap.entrySet()) {
					final int size = entry.getKey();
					final ReferenceQueue<DevicePointerReference> vectorsToRecycleReferenceQueue = entry.getValue();

					Reference<? extends DevicePointerReference> reference;
					while((reference = vectorsToRecycleReferenceQueue.poll()) != null) {
						final CUdeviceptr cuDevicePtr = vectorsInUseReferenceMap.remove(reference);
						if(logger.isLoggable(Level.FINEST)) {
							logger.finest("Freeing device pointer " + cuDevicePtr + " from " + reference);
						}
						try {
							deviceExecutor.submit(new Runnable() {
								@Override
								public void run() {
									cuCtxSynchronize();
									JCudaDriver.cuMemFree(cuDevicePtr);
								}}).get();
						} catch (InterruptedException | ExecutionException e) {
							logger.severe("Unable to free pointer " + cuDevicePtr + " from " + reference);
							throw new RuntimeException(e.getCause());
						}
						deviceAllocMemoryBytes -= size * Sizeof.FLOAT;
					}
				}
			}
		}

		public void purge() {
			System.gc();
			System.runFinalization();
			clean();
			logger.info("Cuda vectors in use: " + vectorsInUseReferenceMap.size() + ". Available device memory: " + getDeviceFreeMemPercentage()*100 + "%");
		}

		/**
		 *
		 * @return Returns the (estimated) percentage amount of free memory on the device.
		 */
		private static float getDeviceFreeMemPercentage() {
			float freeRate;// = 1.0f - 1.1f * (float)deviceAllocMemoryBytes / (float)deviceMaxMemoryBytes;
			try {
				freeRate = deviceExecutor.submit(new Callable<Float>() { @Override
					public Float call() {
					final long[] free = new long[1];
					final long[] total = new long[1];
					jcuda.runtime.JCuda.cudaMemGetInfo(free, total);
					final float freeRate = ((float)free[0]/(total[0]));
					return freeRate;
				}}).get();
			} catch (InterruptedException | ExecutionException e) {
				return freeRate = 0;
			}
			//			System.out.println("Cuda: " + deviceMemoryPool.vectorsInUseReferenceMap.size() + "\t" + freeRate);
			return freeRate;
		}

		/**
		 * Create a vector on device and copy host vector to it.
		 *
		 * @param values Host vector.
		 * @return Pointer to device vector.
		 */
		public DevicePointerReference getDevicePointer(final float[] values) {
			final DevicePointerReference devicePointerReference = getDevicePointer(values.length);
			try {
				deviceExecutor.submit(new Runnable() { @Override
					public void run() {
					cuCtxSynchronize();
					JCudaDriver.cuMemcpyHtoD(devicePointerReference.get(), Pointer.to(values), (long)values.length * Sizeof.FLOAT);
				}}).get();
			} catch (InterruptedException | ExecutionException e) { throw new RuntimeException(e.getCause()); }

			return devicePointerReference;
		}

		public float[] getValuesAsFloat(final DevicePointerReference devicePtr, final int size) {
			final float[] result = new float[size];
			try {
				deviceExecutor.submit(new Runnable() { @Override
					public void run() {
					cuCtxSynchronize();
					cuMemcpyDtoH(Pointer.to(result), devicePtr.get(), size * Sizeof.FLOAT);
					cuCtxSynchronize();
				}}).get();
			} catch (InterruptedException | ExecutionException e) {
				throw new RuntimeException(e.getCause());
			}
			return result;
		}

		public DevicePointerReference callFunctionv1s0(final CUfunction function, final long resultSize, final DevicePointerReference argument1) {
			synchronized (lock) {
				final DevicePointerReference result = getDevicePointer(resultSize);
				callFunction(function, resultSize, new Pointer[] {
						Pointer.to(new int[] { (int)resultSize }),
						Pointer.to(argument1.get()),
						Pointer.to(result.get()) }
						);
				return result;
			}
		}

		public DevicePointerReference callFunctionv2s0(final CUfunction function, final long resultSize, final DevicePointerReference argument1, final DevicePointerReference argument2) {
			synchronized (lock) {
				final DevicePointerReference result = getDevicePointer(resultSize);
				callFunction(function, resultSize, new Pointer[] {
						Pointer.to(new int[] { (int)resultSize }),
						Pointer.to(argument1.get()),
						Pointer.to(argument2.get()),
						Pointer.to(result.get()) }
						);
				return result;
			}
		}

		public DevicePointerReference callFunctionv3s0(final CUfunction function, final long resultSize, final DevicePointerReference argument1, final DevicePointerReference argument2, final DevicePointerReference argument3) {
			synchronized (lock) {
				final DevicePointerReference result = getDevicePointer(resultSize);
				callFunction(function, resultSize, new Pointer[] {
						Pointer.to(new int[] { (int)resultSize }),
						Pointer.to(argument1.get()),
						Pointer.to(argument2.get()),
						Pointer.to(argument3.get()),
						Pointer.to(result.get()) }
						);
				return result;
			}
		}

		public DevicePointerReference callFunctionv1s1(final CUfunction function, final long resultSize, final DevicePointerReference argument1, final double value) {
			synchronized (lock) {
				final DevicePointerReference result = getDevicePointer(resultSize);
				callFunction(function, resultSize, new Pointer[] {
						Pointer.to(new int[] { (int)resultSize }),
						Pointer.to(argument1.get()),
						Pointer.to(new float[] { (float)value }),
						Pointer.to(result.get()) }
						);
				return result;
			}
		}

		public DevicePointerReference callFunctionv2s1(final CUfunction function, final long resultSize, final DevicePointerReference argument1, final DevicePointerReference argument2, final double value) {
			synchronized (lock) {
				final DevicePointerReference result = getDevicePointer(resultSize);
				callFunction(function, resultSize, new Pointer[] {
						Pointer.to(new int[] { (int)resultSize }),
						Pointer.to(argument1.get()),
						Pointer.to(argument2.get()),
						Pointer.to(new float[] { (float)value }),
						Pointer.to(result.get()) }
						);
				return result;
			}
		}

		public void callFunction(final CUfunction function, final long resultSize, final Pointer[] arguments) {
			final int blockSizeX = 1024;
			final int gridSizeX = (int)Math.ceil((double)resultSize / blockSizeX);
			callFunction(function, arguments, gridSizeX, blockSizeX, 0);
		}

		public void callFunction(final CUfunction function, final Pointer[] arguments, final int gridSizeX, final int blockSizeX, final int sharedMemorySize) {
			// Set up the kernel parameters: A pointer to an array
			// of pointers which point to the actual values.
			final Pointer kernelParameters = Pointer.to(arguments);

			deviceExecutor.submit(new Runnable() { @Override
				public void run() {
				//cuCtxSynchronize();
				// Launching on the same stream (default stream)
				cuLaunchKernel(function,
						gridSizeX,  1, 1,      // Grid dimension
						blockSizeX, 1, 1,      // Block dimension
						sharedMemorySize * Sizeof.FLOAT, null,               // Shared memory size and stream
						kernelParameters, null // Kernel- and extra parameters
						);
			}});
		}
	}

	private static DeviceMemoryPool deviceMemoryPool = new DeviceMemoryPool();

	private static final long serialVersionUID = 7620120320663270600L;

	private final double      time;	                // Time (filtration)

	private static final int typePriorityDefault = 20;

	private final int typePriority;

	// Data model for the stochastic case (otherwise null)
	private final DevicePointerReference	realizations;           // Realizations
	private final long			size;

	// Data model for the non-stochastic case (if realizations==null)
	private final double      valueIfNonStochastic;


	private static final Logger logger = Logger.getLogger("net.finmath");

	private static final ExecutorService deviceExecutor = Executors.newSingleThreadExecutor();
	public static final CUdevice device = new CUdevice();
	public static final CUcontext context = new CUcontext();
	public static final CUmodule module = new CUmodule();

	private static final CUfunction capByScalar = new CUfunction();
	private static final CUfunction floorByScalar = new CUfunction();
	private static final CUfunction addScalar = new CUfunction();
	private static final CUfunction subScalar = new CUfunction();
	private static final CUfunction busScalar = new CUfunction();
	private static final CUfunction multScalar = new CUfunction();
	private static final CUfunction divScalar = new CUfunction();
	private static final CUfunction vidScalar = new CUfunction();
	private static final CUfunction cuPow = new CUfunction();
	private static final CUfunction cuSqrt = new CUfunction();
	private static final CUfunction cuExp = new CUfunction();
	private static final CUfunction cuLog = new CUfunction();
	private static final CUfunction invert = new CUfunction();
	private static final CUfunction cuAbs = new CUfunction();
	private static final CUfunction cap = new CUfunction();
	private static final CUfunction cuFloor = new CUfunction();
	private static final CUfunction add = new CUfunction();
	private static final CUfunction sub = new CUfunction();
	private static final CUfunction mult = new CUfunction();
	private static final CUfunction cuDiv = new CUfunction();
	private static final CUfunction accrue = new CUfunction();
	private static final CUfunction discount = new CUfunction();
	private static final CUfunction addProduct = new CUfunction();
	private static final CUfunction addProduct_vs = new CUfunction();		// add the product of a vector and a scalar
	private static final CUfunction reducePartial = new CUfunction();
	private static final CUfunction reduceFloatVectorToDoubleScalar = new CUfunction();

	private static final int reduceGridSize = 1024;

	// Initalize cuda
	static {
		synchronized (deviceMemoryPool) {
			try {
				// Enable exceptions and omit all subsequent error checks
				JCudaDriver.setExceptionsEnabled(true);
				JCudaDriver.setLogLevel(LogLevel.LOG_DEBUG);

				// Create the PTX file by calling the NVCC
				String ptxFileName = null;
				try {
					final URL cuFileURL = RandomVariableCuda.class.getClassLoader().getResource("net/finmath/montecarlo/RandomVariableCudaKernel.cu");
					ptxFileName = net.finmath.jcuda.JCudaUtils.preparePtxFile(cuFileURL);
				} catch (IOException | URISyntaxException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

				final String ptxFileName2 = ptxFileName;
				deviceExecutor.submit(new Runnable() { @Override
					public void run() {
					// Initialize the driver and create a context for the first device.
					cuInit(0);
					cuDeviceGet(device, 0);
					//				cuCtxCreate(context, jcuda.driver.CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC, device);
					cuCtxCreate(context, jcuda.driver.CUctx_flags.CU_CTX_SCHED_AUTO, device);

					// Load the ptx file.
					cuModuleLoad(module, ptxFileName2);

					// Obtain a function pointers
					cuModuleGetFunction(capByScalar, module, "capByScalar");
					cuModuleGetFunction(floorByScalar, module, "floorByScalar");
					cuModuleGetFunction(addScalar, module, "addScalar");
					cuModuleGetFunction(subScalar, module, "subScalar");
					cuModuleGetFunction(busScalar, module, "busScalar");
					cuModuleGetFunction(multScalar, module, "multScalar");
					cuModuleGetFunction(divScalar, module, "divScalar");
					cuModuleGetFunction(vidScalar, module, "vidScalar");
					cuModuleGetFunction(cuPow, module, "cuPow");
					cuModuleGetFunction(cuSqrt, module, "cuSqrt");
					cuModuleGetFunction(cuExp, module, "cuExp");
					cuModuleGetFunction(cuLog, module, "cuLog");
					cuModuleGetFunction(invert, module, "invert");
					cuModuleGetFunction(cuAbs, module, "cuAbs");
					cuModuleGetFunction(cap, module, "cap");
					cuModuleGetFunction(cuFloor, module, "cuFloor");
					cuModuleGetFunction(add, module, "add");
					cuModuleGetFunction(sub, module, "sub");
					cuModuleGetFunction(mult, module, "mult");
					cuModuleGetFunction(cuDiv, module, "cuDiv");
					cuModuleGetFunction(accrue, module, "accrue");
					cuModuleGetFunction(discount, module, "discount");
					cuModuleGetFunction(addProduct, module, "addProduct");
					cuModuleGetFunction(addProduct_vs, module, "addProduct_vs");
					cuModuleGetFunction(reducePartial, module, "reducePartial");
					cuModuleGetFunction(reduceFloatVectorToDoubleScalar, module, "reduceFloatVectorToDoubleScalar");


					final long[] free = new long[1];
					final long[] total = new long[1];
					jcuda.runtime.JCuda.cudaMemGetInfo(free, total);
					DeviceMemoryPool.deviceMaxMemoryBytes = total[0];
					DeviceMemoryPool.deviceAllocMemoryBytes = total[0]-free[0];

					Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
						@Override
						public void run() {
							deviceMemoryPool.purge();
							deviceExecutor.shutdown();
							try {
								deviceExecutor.awaitTermination(1, TimeUnit.SECONDS);
							} catch (final InterruptedException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
						}}
							));
				}});
			} catch(final Error er) {};
		}
	}

	/**
	 * Create a <code>RandomVariableCuda</code>.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param realizations A <code>DevicePointerReference</code> referencing a {@link CUdeviceptr} with the given size. Use {@link #getDevicePointer(long)} to create one.
	 * @param size The size of the vector associated with <code>DevicePointerReference</code>.
	 * @param typePriority The priority of this type in construction of result types. See "operator type priority" for details.
	 * @return A new instance of RandomVariableCuda wrapping the given DevicePointerReference.
	 */
	public static RandomVariableCuda of(final double time, final DevicePointerReference realizations, final long size, final int typePriority) {
		final RandomVariableCuda randomVariableCuda = new RandomVariableCuda(time, realizations, size, typePriority);
		return randomVariableCuda;
	}

	/**
	 * Create a <code>RandomVariableCuda</code>.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param realizations A <code>DevicePointerReference</code> referencing a {@link CUdeviceptr} with the given size. Use {@link #getDevicePointer(long)} to create one.
	 * @param size The size of the vector associated with <code>DevicePointerReference</code>.
	 * @return A new instance of RandomVariableCuda wrapping the given DevicePointerReference.
	 */
	public static RandomVariableCuda of(final double time, final DevicePointerReference realizations, final long size) {
		final RandomVariableCuda randomVariableCuda = new RandomVariableCuda(time, realizations, size, typePriorityDefault);
		return randomVariableCuda;
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param value the value, a constant.
	 */
	public static RandomVariableCuda of(final double time, final double value) {
		final RandomVariableCuda randomVariableCuda = new RandomVariableCuda(time, value, typePriorityDefault);
		return randomVariableCuda;
	}

	/**
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param realizations A <code>DevicePointerReference</code> referencing a {@link CUdeviceptr} with the given size. Use {@link #getDevicePointer(long)} to create one.
	 * @param size The size of the vector associated with <code>DevicePointerReference</code>.
	 * @param typePriority The priority of this type in construction of result types. See "operator type priority" for details.
	 */
	private RandomVariableCuda(final double time, final DevicePointerReference realizations, final long size, final int typePriority) {
		this.time = time;
		this.realizations = realizations;
		this.size = size;
		this.valueIfNonStochastic = Double.NaN;
		this.typePriority = typePriority;
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param value the value, a constant.
	 */
	public RandomVariableCuda(final double value) {
		this(-Double.MAX_VALUE, value);
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param value the value, a constant.
	 * @param typePriority The priority of this type in construction of result types. See "operator type priority" for details.
	 */
	public RandomVariableCuda(final double time, final double value, final int typePriority) {
		this.time = time;
		this.realizations = null;
		this.size = 1;
		this.valueIfNonStochastic = value;
		this.typePriority = typePriority;
	}

	/**
	 * Create a stochastic random variable.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param realisations the vector of realizations.
	 * @param typePriority The priority of this type in construction of result types. See "operator type priority" for details.
	 */
	public RandomVariableCuda(final double time, final float[] realisations, final int typePriority) {
		this(time, getDevicePointer(realisations), realisations.length, typePriority);
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param value the value, a constant.
	 */
	public RandomVariableCuda(final double time, final double value) {
		this(time, value, typePriorityDefault);
	}

	/**
	 * Create a stochastic random variable.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param realisations the vector of realizations.
	 */
	public RandomVariableCuda(final double time, final float[] realisations) {
		this(time, getDevicePointer(realisations), realisations.length, typePriorityDefault);
	}

	/**
	 * Create a stochastic random variable.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param realisations the vector of realizations.
	 */
	public RandomVariableCuda(final double time, final double[] realisations) {
		this(time, getFloatArray(realisations));
	}

	/**
	 * Create a stochastic random variable.
	 *
	 * @param realisations the vector of realizations.
	 */
	public RandomVariableCuda(final float[] realisations) {
		this(0.0, realisations);
	}


	public static DevicePointerReference getDevicePointer(final long size) {
		return deviceMemoryPool.getDevicePointer(size);
	}

	/**
	 * Create a vector on device and copy host vector to it.
	 *
	 * @param values Host vector.
	 * @return Pointer to device vector.
	 */
	private static DevicePointerReference getDevicePointer(final float[] values) {
		return deviceMemoryPool.getDevicePointer(values);
	}

	public static void clean() {
		deviceMemoryPool.clean();
	}

	public static void purge() {
		deviceMemoryPool.purge();
	}

	private static RandomVariableCuda getRandomVariableCuda(final RandomVariable randomVariable) {
		if(randomVariable instanceof RandomVariableCuda) {
			return (RandomVariableCuda)randomVariable;
		} else {
			final RandomVariableCuda randomVariableCuda = new RandomVariableCuda(randomVariable.getFiltrationTime(), randomVariable.getRealizations());
			return randomVariableCuda;
		}
	}

	private static float[] getFloatArray(final double[] arrayOfDouble) {
		final float[] arrayOfFloat = new float[arrayOfDouble.length];
		for(int i=0; i<arrayOfDouble.length; i++) {
			arrayOfFloat[i] = (float)arrayOfDouble[i];
		}
		return arrayOfFloat;
	}

	private static double[] getDoubleArray(final float[] arrayOfFloat) {
		final double[] arrayOfDouble = new double[arrayOfFloat.length];
		for(int i=0; i<arrayOfFloat.length; i++) {
			arrayOfDouble[i] = arrayOfFloat[i];
		}
		return arrayOfDouble;
	}

	@Override
	public boolean equals(final RandomVariable randomVariable) {
		throw new UnsupportedOperationException();
		/*
		if(this.time != randomVariable.getFiltrationTime()) return false;
		if(this.isDeterministic() && randomVariable.isDeterministic()) {
			return this.valueIfNonStochastic == randomVariable.doubleValue();
		}

		if(this.isDeterministic() != randomVariable.isDeterministic()) return false;

		for(int i=0; i<realizations.length; i++) if(realizations[i] != randomVariable.get(i)) return false;

		return true;
		 */
	}

	@Override
	public double getFiltrationTime() {
		return time;
	}

	@Override
	public int getTypePriority() {
		return typePriority;
	}

	@Override
	public double get(final int pathOrState) {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		} else {
			throw new UnsupportedOperationException();
		}
	}

	@Override
	public int size() {
		if(isDeterministic()) {
			return 1;
		} else {
			return (int)this.size;
		}
	}

	@Override
	public double getMin() {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		}

		final double[] realizations = getRealizations();

		// TODO: Use kernel
		double min = Double.MAX_VALUE;
		if(realizations.length != 0)
		{
			min = realizations[0];     /// @see getMax()
		}
		for(int i=0; i<realizations.length; i++) {
			min = Math.min(realizations[i],min);
		}
		return min;
	}

	@Override
	public double getMax() {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		}

		final double[] realizations = getRealizations();

		// TODO: Use kernel
		double max = -Double.MAX_VALUE;
		if(realizations.length != 0) {
			max = realizations[0];
		}
		for(int i=0; i<realizations.length; i++) {
			max = Math.max(realizations[i],max);
		}
		return max;
	}

	@Override
	public double getAverage() {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		// TODO: Use kernel
		return (new RandomVariableFromFloatArray(getFiltrationTime(), deviceMemoryPool.getValuesAsFloat(realizations, size()))).getAverage();

		//RandomVariable reduced = reduceToDouble();
		//return reduced.getAverage() * reduced.size() / size();		// Temp hack @FIXME @TODO
		//return  reduce()/size();
	}

	@Override
	public double getAverage(final RandomVariable probabilities) {
		return this.mult(probabilities).getAverage();
	}

	@Override
	public double getVariance() {
		if(isDeterministic()) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		final double average = getAverage();
		return this.squared().getAverage() - average*average;
	}

	@Override
	public double getVariance(final RandomVariable probabilities) {
		final double average = getAverage(probabilities);
		return this.squared().sub(average*average).getAverage(probabilities);
	}

	@Override
	public double getSampleVariance() {
		if(isDeterministic() || size() == 1) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		return getVariance() * size()/(size()-1);
	}

	@Override
	public double getStandardDeviation() {
		if(isDeterministic()) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		return Math.sqrt(getVariance());
	}

	@Override
	public double getStandardDeviation(final RandomVariable probabilities) {
		if(isDeterministic()) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		return Math.sqrt(getVariance(probabilities));
	}

	@Override
	public double getStandardError() {
		if(isDeterministic()) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		return getStandardDeviation()/Math.sqrt(size());
	}

	@Override
	public double getStandardError(final RandomVariable probabilities) {
		if(isDeterministic()) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		return getStandardDeviation(probabilities)/Math.sqrt(size());
	}

	@Override
	public double getQuantile(final double quantile) {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		final double[] realizations = getRealizations();

		final double[] realizationsSorted = realizations;
		java.util.Arrays.sort(realizationsSorted);

		final int indexOfQuantileValue = Math.min(Math.max((int)Math.round((size()+1) * (1-quantile) - 1), 0), size()-1);

		return realizationsSorted[indexOfQuantileValue];
	}

	@Override
	public double getQuantile(final double quantile, final RandomVariable probabilities) {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		throw new RuntimeException("Method not implemented.");
	}

	@Override
	public double getQuantileExpectation(final double quantileStart, final double quantileEnd) {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		}
		if(size() == 0) {
			return Double.NaN;
		}
		if(quantileStart > quantileEnd) {
			return getQuantileExpectation(quantileEnd, quantileStart);
		}

		final double[] realizationsSorted = getRealizations();
		java.util.Arrays.sort(realizationsSorted);

		final int indexOfQuantileValueStart	= Math.min(Math.max((int)Math.round((size()+1) * quantileStart - 1), 0), size()-1);
		final int indexOfQuantileValueEnd		= Math.min(Math.max((int)Math.round((size()+1) * quantileEnd - 1), 0), size()-1);

		double quantileExpectation = 0.0;
		for (int i=indexOfQuantileValueStart; i<=indexOfQuantileValueEnd;i++) {
			quantileExpectation += realizationsSorted[i];
		}
		quantileExpectation /= indexOfQuantileValueEnd-indexOfQuantileValueStart+1;

		return quantileExpectation;
	}

	@Override
	public double[] getHistogram(final double[] intervalPoints)
	{
		final double[] histogramValues = new double[intervalPoints.length+1];

		if(isDeterministic()) {
			java.util.Arrays.fill(histogramValues, 0.0);
			for (int intervalIndex=0; intervalIndex<intervalPoints.length; intervalIndex++)
			{
				if(valueIfNonStochastic > intervalPoints[intervalIndex]) {
					histogramValues[intervalIndex] = 1.0;
					break;
				}
			}
			histogramValues[intervalPoints.length] = 1.0;
		}
		else {
			final double[] realizationsSorted = getRealizations();
			java.util.Arrays.sort(realizationsSorted);

			int sampleIndex=0;
			for (int intervalIndex=0; intervalIndex<intervalPoints.length; intervalIndex++)
			{
				int sampleCount = 0;
				while (sampleIndex < realizationsSorted.length &&
						realizationsSorted[sampleIndex] <= intervalPoints[intervalIndex])
				{
					sampleIndex++;
					sampleCount++;
				}
				histogramValues[intervalIndex] = sampleCount;
			}
			histogramValues[intervalPoints.length] = realizationsSorted.length-sampleIndex;

			// Normalize histogramValues
			if(realizationsSorted.length > 0) {
				for(int i=0; i<histogramValues.length; i++) {
					histogramValues[i] /= realizationsSorted.length;
				}
			}
		}

		return histogramValues;
	}

	@Override
	public double[][] getHistogram(final int numberOfPoints, final double standardDeviations) {
		final double[] intervalPoints = new double[numberOfPoints];
		final double[] anchorPoints	= new double[numberOfPoints+1];
		final double center	= getAverage();
		final double radius	= standardDeviations * getStandardDeviation();
		final double stepSize	= (numberOfPoints-1) / 2.0;
		for(int i=0; i<numberOfPoints;i++) {
			final double alpha = (-(double)(numberOfPoints-1) / 2.0 + i) / stepSize;
			intervalPoints[i]	= center + alpha * radius;
			anchorPoints[i]		= center + alpha * radius - radius / (2 * stepSize);
		}
		anchorPoints[numberOfPoints] = center + 1 * radius + radius / (2 * stepSize);

		final double[][] result = new double[2][];
		result[0] = anchorPoints;
		result[1] = getHistogram(intervalPoints);

		return result;
	}

	@Override
	public boolean isDeterministic() {
		return realizations == null;
	}

	@Override
	public RandomVariable cache() {
		return this;
		/*
		final float[] values = new float[(int)size];
		try {
			deviceExecutor.submit(new Runnable() { public void run() {
				cuCtxSynchronize();
				cuMemcpyDtoH(Pointer.to(values), realizations, size * Sizeof.FLOAT);
				cuCtxSynchronize();
			}}).get();
		} catch (InterruptedException | ExecutionException e) { throw new RuntimeException(e.getCause()); }
		return new RandomVariableFromFloatArray(time, values);
		 */
	}

	@Override
	public double[] getRealizations() {
		if(isDeterministic()) {
			final double[] result = new double[] { valueIfNonStochastic };
			return result;
		} else {
			return getDoubleArray(deviceMemoryPool.getValuesAsFloat(realizations, size()));
		}
	}

	@Override
	public Double doubleValue() {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		} else {
			throw new UnsupportedOperationException("The random variable is non-deterministic");
		}
	}

	@Override
	public IntToDoubleFunction getOperator() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DoubleStream getRealizationsStream() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariable apply(final DoubleUnaryOperator function) {
		throw new UnsupportedOperationException();
		/*
		if(isDeterministic()) {
			double newValueIfNonStochastic = function.value(valueIfNonStochastic);
			return of(time, newValueIfNonStochastic);
		}
		else {
			double[] newRealizations = new double[realizations.length];
			for(int i=0; i<newRealizations.length; i++) newRealizations[i]		 = function.value(realizations[i]);
			return of(time, newRealizations);
		}
		 */
	}

	@Override
	public RandomVariable apply(final DoubleBinaryOperator operator, final RandomVariable argument) {
		throw new UnsupportedOperationException();
	}

	@Override
	public RandomVariable apply(final DoubleTernaryOperator operator, final RandomVariable argument1, final RandomVariable argument2) {
		throw new UnsupportedOperationException();
	}

	@Override
	public RandomVariable cap(final double cap) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.min(valueIfNonStochastic,cap);
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s1(capByScalar, size, realizations, cap);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable floor(final double floor) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.max(valueIfNonStochastic,floor);
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s1(floorByScalar, size, realizations, floor);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable add(final double value) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic + value;
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s1(addScalar, size, realizations, value);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable sub(final double value) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic - value;
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s1(subScalar, size, realizations, value);
			return of(time, result, size());
		}
	}

	public RandomVariable bus(final double value) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = -valueIfNonStochastic + value;
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s1(busScalar, size, realizations, value);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable mult(final double value) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic * value;
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s1(multScalar, size, realizations, value);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable div(final double value) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic / value;
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s1(divScalar, size, realizations, value);
			return of(time, result, size());
		}
	}

	public RandomVariable vid(final double value) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = value / valueIfNonStochastic;
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s1(vidScalar, size, realizations, value);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable pow(final double exponent) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.pow(valueIfNonStochastic,exponent);
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s1(cuPow, size, realizations, exponent);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable average() {
		return of(-Double.MAX_VALUE, getAverage());
	}

	@Override
	public RandomVariable squared() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic * valueIfNonStochastic;
			return of(time, newValueIfNonStochastic);
		} else {
			return this.mult(this);
		}
	}

	@Override
	public RandomVariable sqrt() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.sqrt(valueIfNonStochastic);
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s0(cuSqrt, size, realizations);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable invert() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = 1.0/valueIfNonStochastic;
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s0(invert, size, realizations);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable abs() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.abs(valueIfNonStochastic);
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s0(cuAbs, size, realizations);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable exp() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.exp(valueIfNonStochastic);
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s0(cuExp, size, realizations);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable log() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.log(valueIfNonStochastic);
			return of(time, newValueIfNonStochastic);
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv1s0(cuLog, size, realizations);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable sin() {
		throw new UnsupportedOperationException();
		/*
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.sin(valueIfNonStochastic);
			return of(time, newValueIfNonStochastic);
		}
		else {
			double[] newRealizations = new double[realizations.length];
			for(int i=0; i<newRealizations.length; i++) newRealizations[i]		 = Math.sin(realizations[i]);
			return of(time, newRealizations);
		}
		 */
	}

	@Override
	public RandomVariable cos() {
		throw new UnsupportedOperationException();
		/*
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.cos(valueIfNonStochastic);
			return of(time, newValueIfNonStochastic);
		}
		else {
			double[] newRealizations = new double[realizations.length];
			for(int i=0; i<newRealizations.length; i++) newRealizations[i]		 = Math.cos(realizations[i]);
			return of(time, newRealizations);
		}
		 */
	}

	/*
	 * Binary operators: checking for return type priority.
	 */

	@Override
	public RandomVariable add(final RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.add(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic + randomVariable.doubleValue();
			return of(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return getRandomVariableCuda(randomVariable).add(valueIfNonStochastic);
		} else if(randomVariable.isDeterministic()) {
			return this.add(randomVariable.doubleValue());
		} else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv2s0(add, size, realizations, getRandomVariableCuda(randomVariable).realizations);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable sub(final RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.bus(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic - randomVariable.doubleValue();
			return of(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return getRandomVariableCuda(randomVariable).bus(valueIfNonStochastic);
		} else if(randomVariable.isDeterministic()) {
			return this.sub(randomVariable.doubleValue());
		} else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv2s0(sub, size, realizations, getRandomVariableCuda(randomVariable).realizations);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable bus(final RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.sub(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = -valueIfNonStochastic + randomVariable.doubleValue();
			return of(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return getRandomVariableCuda(randomVariable).sub(valueIfNonStochastic);
		} else if(randomVariable.isDeterministic()) {
			return this.bus(randomVariable.doubleValue());
		} else {
			// flipped arguments
			final DevicePointerReference result = deviceMemoryPool.callFunctionv2s0(sub, size, getRandomVariableCuda(randomVariable).realizations, realizations);
			return of(time, result, size());
		}
	}

	@Override
	public RandomVariable mult(final RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.mult(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic * randomVariable.doubleValue();
			return of(newTime, newValueIfNonStochastic);
		}
		else if(randomVariable.isDeterministic()) {
			return this.mult(randomVariable.doubleValue());
		} else if(isDeterministic() && !randomVariable.isDeterministic()) {
			return getRandomVariableCuda(randomVariable).mult(this.valueIfNonStochastic);
		} else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv2s0(mult, size, realizations, getRandomVariableCuda(randomVariable).realizations);
			return of(newTime, result, size());
		}
	}

	@Override
	public RandomVariable div(final RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.vid(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic / randomVariable.doubleValue();
			return of(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return getRandomVariableCuda(randomVariable).vid(valueIfNonStochastic);
		} else if(randomVariable.isDeterministic()) {
			return this.div(randomVariable.doubleValue());
		} else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv2s0(cuDiv, size, realizations, getRandomVariableCuda(randomVariable).realizations);
			return of(newTime, result, size());
		}
	}

	@Override
	public RandomVariable vid(final RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.vid(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = randomVariable.doubleValue() / valueIfNonStochastic;
			return of(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return getRandomVariableCuda(randomVariable).div(valueIfNonStochastic);
		} else if(randomVariable.isDeterministic()) {
			return this.vid(randomVariable.doubleValue());
		} else {
			// flipped arguments
			final DevicePointerReference result = deviceMemoryPool.callFunctionv2s0(cuDiv, size, getRandomVariableCuda(randomVariable).realizations, realizations);
			return of(newTime, result, size());
		}
	}

	@Override
	public RandomVariable cap(final RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.cap(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = Math.min(valueIfNonStochastic, randomVariable.doubleValue());
			return of(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return randomVariable.cap(valueIfNonStochastic);
		} else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv2s0(cap, size, realizations, getRandomVariableCuda(randomVariable).realizations);
			return of(newTime, result, size());
		}
	}

	@Override
	public RandomVariable floor(final RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.floor(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = Math.max(valueIfNonStochastic, randomVariable.doubleValue());
			return of(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return getRandomVariableCuda(randomVariable).floor(valueIfNonStochastic);
		} else if(randomVariable.isDeterministic()) {
			return this.floor(randomVariable.doubleValue());
		} else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv2s0(cuFloor, size, realizations, getRandomVariableCuda(randomVariable).realizations);
			return of(newTime, result, size());
		}
	}

	@Override
	public RandomVariable accrue(final RandomVariable rate, final double periodLength) {
		if(rate.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return rate.mult(periodLength).add(1.0).mult(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, rate.getFiltrationTime());

		if(rate.isDeterministic()) {
			return this.mult(1.0 + rate.doubleValue() * periodLength);
		} else if(isDeterministic() && !rate.isDeterministic()) {
			return getRandomVariableCuda(rate.mult(periodLength).add(1.0).mult(valueIfNonStochastic));
		} else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv2s1(accrue, size, realizations, getRandomVariableCuda(rate).realizations, periodLength);
			return of(newTime, result, size());
		}
	}

	@Override
	public RandomVariable discount(final RandomVariable rate, final double periodLength) {
		if(rate.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return rate.mult(periodLength).add(1.0).invert().mult(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, rate.getFiltrationTime());

		if(rate.isDeterministic()) {
			return this.div(1.0 + rate.doubleValue() * periodLength);
		} else if(isDeterministic() && !rate.isDeterministic()) {
			if(valueIfNonStochastic == 0) {
				return this;
			}
			return (getRandomVariableCuda(rate.mult(periodLength).add(1.0)).vid(valueIfNonStochastic));
		}
		else {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv2s1(discount, size, realizations, getRandomVariableCuda(rate).realizations, periodLength);
			return of(newTime, result, size());
		}
	}

	/*
	 * Ternary operators: checking for return type priority.
	 * @TODO add checking for return type priority.
	 */

	@Override
	public RandomVariable choose(final RandomVariable valueIfTriggerNonNegative, final RandomVariable valueIfTriggerNegative) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariable addProduct(final RandomVariable factor1, final double factor2) {
		if(factor1.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return factor1.mult(factor2).add(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, factor1.getFiltrationTime());

		if(factor1.isDeterministic()) {
			return this.add(factor1.doubleValue() * factor2);
		} else if(!isDeterministic() && !factor1.isDeterministic()) {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv2s1(addProduct_vs, size, realizations, getRandomVariableCuda(factor1).realizations, factor2);
			return of(newTime, result, size());
		} else {
			return this.add(factor1.mult(factor2));
		}
	}

	@Override
	public RandomVariable addProduct(final RandomVariable factor1, final RandomVariable factor2) {
		if(factor1.getTypePriority() > this.getTypePriority() || factor2.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return factor1.mult(factor2).add(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(Math.max(time, factor1.getFiltrationTime()), factor2.getFiltrationTime());

		if(isDeterministic() && factor1.isDeterministic() && factor2.isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic + (factor1.doubleValue() * factor2.doubleValue());
			return of(newTime, newValueIfNonStochastic);
		}
		else if(factor1.isDeterministic() && factor2.isDeterministic()) {
			return add(factor1.doubleValue() * factor2.doubleValue());
		} else if(factor2.isDeterministic()) {
			return this.addProduct(factor1, factor2.doubleValue());
		} else if(factor1.isDeterministic()) {
			return this.addProduct(factor2, factor1.doubleValue());
		} else if(!isDeterministic() && !factor1.isDeterministic() && !factor2.isDeterministic()) {
			final DevicePointerReference result = deviceMemoryPool.callFunctionv3s0(addProduct, size, realizations, getRandomVariableCuda(factor1).realizations, getRandomVariableCuda(factor2).realizations);
			return of(newTime, result, size());
		} else {
			return this.add(factor1.mult(factor2));
		}
	}

	@Override
	public RandomVariable addRatio(final RandomVariable numerator, final RandomVariable denominator) {
		// TODO Implement a kernel here
		return this.add(numerator.div(denominator));
	}

	@Override
	public RandomVariable subRatio(final RandomVariable numerator, final RandomVariable denominator) {
		// TODO Implement a kernel here
		return this.sub(numerator.div(denominator));
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariable#isNaN()
	 */
	@Override
	public RandomVariable isNaN() {
		// TODO Auto-generated method stub
		return null;
	}

	/*
	 * Cuda specific implementations
	 */


	private RandomVariableFromDoubleArray reduceToDouble() {

		final int blockSizeX = reduceGridSize;
		final int gridSizeX = (int)Math.ceil((double)size()/2 / blockSizeX);

		final DevicePointerReference reduceVector = getDevicePointer(2*gridSizeX);

		deviceMemoryPool.callFunction(reduceFloatVectorToDoubleScalar, new Pointer[] {
				Pointer.to(new int[] { size() }),
				Pointer.to(realizations.get()),
				Pointer.to(reduceVector.get())},
				gridSizeX, blockSizeX, blockSizeX*2*3);

		final double[] result = new double[gridSizeX];
		try {
			deviceExecutor.submit(new Runnable() { @Override
				public void run() {
				cuCtxSynchronize();
				cuMemcpyDtoH(Pointer.to(result), reduceVector.get(), gridSizeX * Sizeof.DOUBLE);
				cuCtxSynchronize();
			}}).get();
		} catch (InterruptedException | ExecutionException e) {
			throw new RuntimeException(e.getCause());
		}

		return (new RandomVariableFromDoubleArray(time, result));
	}

	private double reduce() {
		if(this.isDeterministic()) {
			return valueIfNonStochastic;
		}

		RandomVariableCuda reduced = this;
		while(reduced.size() > 1) {
			reduced = reduced.reduceBySize(reduceGridSize);
		}
		return reduced.getRealizations()[0];
	}

	private RandomVariableCuda reduceBySize(final int bySize) {
		final int blockSizeX = bySize;
		final int gridSizeX = (int)Math.ceil((double)size()/2 / blockSizeX);
		final DevicePointerReference reduceVector = getDevicePointer(gridSizeX);

		deviceMemoryPool.callFunction(reducePartial, new Pointer[] {
				Pointer.to(new int[] { size() }),
				Pointer.to(realizations.get()),
				Pointer.to(reduceVector.get())},
				gridSizeX, blockSizeX, blockSizeX*2*3);

		return of(-Double.MAX_VALUE, reduceVector, gridSizeX);
	}
}