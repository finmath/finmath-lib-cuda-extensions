package net.finmath.jcuda;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.logging.Logger;

import org.apache.commons.io.IOUtils;

/**
 * Adapted from JCuda examples: Reads a CUDA file, compiles it to a PTX file
 * using NVCC, loads the PTX file as a module and executes
 * the kernel function.
 */
public class JCudaUtils
{
	private static Logger logger = Logger.getLogger("net.finmath");

	/**
	 * The extension of the given file name is replaced with "ptx".
	 * If the file with the resulting name does not exist, it is
	 * compiled from the given file using NVCC. The name of the
	 * PTX file is returned.
	 *
	 * @param cuFileStream The name of the .cu file
	 * @param arch String specifying the architecture (see nvcc command, -arch argument). Examples are sm_12, sm_20, sm_30
	 * @return The name of the PTX file.
	 * @throws IOException Thrown if an I/O error occurs.
	 * @throws URISyntaxException Thrown if the cuFileURL cannot be converted to an URI.
	 */
	public static String preparePtxFile(final InputStream cuFileStream, final String arch) throws IOException, URISyntaxException
	{
		//		final String cuFileName = Paths.get(cuFileURI).toFile().getAbsolutePath();
		final Path file = Files.createTempFile("RandomVariableCudaKernel", ".cu");
		Files.copy(cuFileStream, file, StandardCopyOption.REPLACE_EXISTING);
		final String cuFileName = file.toFile().getAbsolutePath();

		int endIndex = cuFileName.lastIndexOf('.');
		if (endIndex == -1)
		{
			endIndex = cuFileName.length()-1;
		}
		final String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
		final File ptxFile = new File(ptxFileName);
		if (ptxFile.exists()) {
			return ptxFileName;
		}

		final File cuFile = new File(cuFileName);
		if (!cuFile.exists()) {
			throw new IOException("Input file not found: "+cuFileName);
		}

		/*
		 * Check for 64 bit or 32 bit
		 */
		final String modelString = "-m"+System.getProperty("sun.arch.data.model");

		final String[] command = {
				"nvcc",
				"-arch",
				arch,
				"-fmad",
				"false",
				modelString,
				"-ptx",
				cuFile.getPath(),
				"-o",
				ptxFileName };

		//		String command = "nvcc " + modelString + " -ptx " + "" + cuFile.getPath() + " -o " + ptxFileName;

		logger.info("Executing\n"+Arrays.toString(command));
		final Process process = Runtime.getRuntime().exec(command);

		final String errorMessage = IOUtils.toString(process.getErrorStream(), Charset.defaultCharset());
		final String outputMessage = IOUtils.toString(process.getInputStream(), Charset.defaultCharset());
		int exitValue = 0;
		try
		{
			exitValue = process.waitFor();
		}
		catch (final InterruptedException e)
		{
			Thread.currentThread().interrupt();
			throw new IOException("Interrupted while waiting for nvcc output", e);
		}

		if (exitValue != 0)
		{
			logger.severe("nvcc process exitValue "+ exitValue +
					"\nerrorMessage: "+errorMessage + 
					"\noutputMessage: "+outputMessage);

			throw new IOException("Could not create .ptx file: "+errorMessage);
		}

		logger.info("Finished creating PTX file");

		return ptxFileName;
	}
}