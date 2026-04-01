## Curve fitting methods

### QurvE approach

- based on approach in QurvE paper, which based it on the approach laid out in
  grofit R-package. See their
  [README](https://github.com/NicWir/QurvE?tab=readme-ov-file#growth-profiling-methods)
  and
  [Kahm et.al. 2014](https://www.jstatsoft.org/article/view/v033i07)
- cubic smoothing spline from R was used:
  [docs](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/smooth.spline)

- the curves were fitted on the log-transformed data of the plots, shifted by the minimal
  value

### This app

In this app we use the [growthcurves package](https://growthcurves.readthedocs.io/) we
developed, which is highly inspired by the review article on fitting growth curves:

> Ghenu, A.-H., Marrec, L. & Bank, C. Challenges and pitfalls of inferring microbial
> growth rates from lab cultures. Front. Ecol. Evol. 11, 1313500 (2024).
> https://doi.org/10.3389/fevo.2023.1313500

We allow the application of parametric, phenomological and non-parametric models, on
the filtered and smoothed data, with the option applied on the upload page:

- the rolling median was used to pre-smooth the data before fitting the splines
- growthcurves fits a parametric model to the data or fits equations on the
  log transformed data.
