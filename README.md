About finmath lib cuda extensions
==========

****************************************

**Enabling finmath lib with cuda via jcuda.**

****************************************

The finmath lib cuda extension provide an implementation of the finmath lib
interfaces
	RandomVariableInterface
and
	BrownianMotionInterface
compatible with finmath lib 2.4.3 or later.

Since object of type BrownianMotionInterface are taking the role of a factory for objects
of type RandomVariableInterface, injecting the BrownianMotionCuda will result in most
finmath-lib models performing their calculations on the GPU - seamlessly.

References
--------------------------------------

-   [finmath lib Project documentation][]  
    provides the documentation of the library api.
-   [finmath lib API documentation][]  
    provides the documentation of the library api.
-   [finmath.net special topics][]  
    cover some selected topics with demo spreadsheets and uml diagrams.
    Some topics come with additional documentations (technical papers).


License
-------

The code of "finmath lib", "finmath experiments" and "finmath lib cuda extensions" (packages
`net.finmath.*`) are distributed under the [Apache License version
2.0][], unless otherwise explicitly stated.
 

  [finmath lib Project documentation]: http://finmath.net/finmath-lib/ 
  [finmath lib API documentation]: http://finmath.net/finmath-lib/apidocs/
  [finmath.net special topics]: http://www.finmath.net/topics
  [Apache License version 2.0]: http://www.apache.org/licenses/LICENSE-2.0.html
