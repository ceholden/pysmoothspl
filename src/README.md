# History

Source code extracted from R language so the underlying C
functions can be run from other languages.

Who did what part of this extraction isn't known to me, but this
port to Python could not have been possible without the work of the
Keith Ma and Katia Oleinik BU RCS who brought the code out of R and
shaped it into a compiling and working state.


## Code structure
From R's source src/library/stats/splines-README

```
smooth.spline()  [R]
 |
 \--> rbart() [ = C_rbart() ]           src/qsbart.f
      |
      \--> sbart()                      src/sbart.c
           |--> sgram()                 src/sgram.f
           |    |--> interv()           src/bvalue.f
           |    \--> bsplvd()           src/bsplvd.f
           |         \--> bsplvb()          "
           |--> stxwx()                 src/stxwx.f
           |    |--> interv()             (see above)
           |    \--> bsplvd()               "
           |         \--> bsplvb()          "
           \--> sslvrg()                src/sslvrg.f
                |--> sinerp()           src/sinerp.f
                |--> bvalue()           src/bvalue.f   (above)
                |    \--> interv()        (see above)
                |--> interv()               "
                |--> bsplvd()               "
                |    \--> bsplvb()          "
                |--> dpbfa()            ../../appl/dpbfa.f  {LINPACK}
                \--> dpbsl()            ../../appl/dpbsl.f


predict.smooth.spline()  [R]
 |
 \--> bvalus()                          src/bvalus.f
       \--> bvalue()                    (see above)
             \--> interv()                "
```
