# Impact of COVID-19 restrictions on surface concentrations of nitrogen dioxide (NO2)

Sample code to calculate NO2 concentration anomalies in the wake of the COVID-19 pandemic following the methodology described in Keller et al. (2020). For convenience, preprocessed surface observation data (from https://openaq.org/) and corresponding GEOS-CF model output (available at https://gmao.gsfc.nasa.gov/weather_prediction/GEOS-CF/data_access/) for selected locations are made available at https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/COVID_NO2/examples/.
Tested with Python 3.6.7.

Usage:
`python no2_covid_example.py -c 'NewYork' 'Paris'`

**References:**
Keller, C. A., Evans, M. J., Knowland, K. E., Hasenkopf, C. A., Modekurty, S., Lucchesi, R. A., Oda, T., Franca, B. B., Mandarino, F. C., Díaz Suárez, M. V., Ryan, R. G., Fakes, L. H., and Pawson, S.: Global Impact of COVID-19 Restrictions on the Surface Concentrations of Nitrogen Dioxide and Ozone, Atmos. Chem. Phys. Discuss., https://doi.org/10.5194/acp-2020-685, in review, 2020.

Contact:
christoph.a.keller@nasa.gov
