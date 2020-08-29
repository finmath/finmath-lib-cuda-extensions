
module net.finmath.gpu {
	exports net.finmath.cuda.montecarlo;
	exports net.finmath.cuda.montecarlo.alternative;
	exports net.finmath.opencl.montecarlo;

	requires net.finmath.lib;

	requires java.logging;
	requires org.apache.commons.io;

	requires jcuda;
	requires jcurand;
	requires jocl;
}
