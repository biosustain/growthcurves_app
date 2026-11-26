# Curve fitting methods

## References

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

## Specific growth rate

The specific growth rate $\mu$ is growth per unit of biomass. In exponential growth the
population grows in proportion to its size:

$$
\frac{dN}{dt} = \mu N \quad\Longrightarrow\quad \mu = \frac{1}{N}\frac{dN}{dt} =
\frac{d \ln N}{dt} \qquad [\mathrm{h}^{-1}]
$$

- $dN/dt$ alone is the absolute rate (OD per hour). It rises just because there are more
  cells, even when each cell divides at the same speed.
- Dividing by $N$ removes the effect of population size and leaves the per-cell rate
  $\mu$. That makes $\mu$ comparable across curves and gives the doubling time $t_d = \ln
  2 / \mu_{\max}$.
- During clean exponential growth $\mu$ is flat; its peak is $\mu_{\max}$.

$\frac{1}{N}\frac{dN}{dt}$ and $\frac{d \ln N}{dt}$ are the same quantity mathematically,
but not numerically: dividing by $N$ blows up noise at low OD, while the slope of $\ln N$
does not. A baseline-corrected $N \le 0$ also flips the sign of
$\frac{1}{N}\frac{dN}{dt}$ or makes it explode, so those reads are dropped for any $\mu$
plot.

## Continuous specific growth rate

The pink line in any $\mu$ plot is the slope of $\ln N$ over a window of $w$ reads around
each read ($w$ = `sg_window` = 15). A toggle picks the
method:

1. **Savitzky–Golay** (`savgol_filter(np.log(N), w, p, deriv=1, delta=dt)`): fits a
   polynomial of order $p$ to $\ln N$ in the window and takes its slope, so it smooths
   and differentiates in one step. With $p = 1$ it is exactly the least-squares slope
   over $w$ evenly spaced reads; $p = 2$ also follows curvature, so the peak is less
   flattened. It assumes even spacing ($\Delta t$ = median time step), so gaps from lasso
   selections or dilution cuts distort it.
2. **Sliding window** (`compute_sliding_window_growth_rate`): least-squares line through
   $(t_j, \ln N_j)$ over the window, using the real read times, so it handles uneven time
   spacing and gaps:

$$
\mu_i = \frac{\sum_j (t_j - \bar t)(\ln N_j - \overline{\ln N})}{\sum_j (t_j - \bar
t)^2}, \qquad j \in \left[i - \lfloor w/2 \rfloor,\ i + \lfloor w/2 \rfloor\right]
$$

A wider window means less noise but a flatter, lower peak, so it slightly underestimates
$\mu_{\max}$.

## Where $\mu_{\max}$ comes from

Everything except the pink line in the specific growth rate plot (which you need to
toggle to see) comes from the model selected in the analysis options:

- **$\mu_{\max}$** (tables, markers, doubling time, `min_growth_rate` check): exact from
  the fitted parameters for parametric models where a formula exists, the largest slope
  of the $\ln N$ spline for `spline`, the slope of the best window for `sliding_window`.
- **Blue model line** and the **$\mu$ panel under the whole-run plot**: the fitted
  model's $\mu$ over the fit range. Parametric models use $\frac{1}{N}\frac{dN}{dt}$ of
  the model evaluated at the read times; the spline uses its exact derivative; sliding
  window is a flat line at the fitted slope over the chosen window.
- The $\mu_{\max}$ marker can sit slightly off the model line, because the marker is
  exact while the line is a numerical derivative at the read times.
- The Savitzky–Golay / sliding-window toggle only changes the pink line, not
  $\mu_{\max}$.
