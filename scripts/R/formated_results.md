## Salience x window-time

### Object

Coefficient                          | Median | MAD_SD | p-value
-------------------------------------|--------|--------|---------
(Intercept)                          | 0.5    | 0.2    | 0.012
salience_labellow                    | 0.1    | 0.2    | 0.739
winstart_labellate                   | 0.0    | 0.2    | 0.919
salience_labellow:winstart_labellate | 0.5    | 0.3    | 0.180
(phi)                                | 6.2    | 1.1    | <1e-3

Coefficient                     | Df  | Sum Sq | Mean Sq | F value | Pr(>F)
--------------------------------|-----|--------|---------|---------|---------
salience_label                  | 1   | 0.0027 | 0.00272 | 0.083   | 0.774
winstart_label                  | 1   | 0.0032 | 0.00324 | 0.099   | 0.753
salience_label:winstart_label   | 1   | 0.0166 | 0.01660 | 0.508   | 0.478
Residuals                       | 80  | 2.6123 | 0.03265 |         |

### Spatial

Coefficient                          | Median | MAD_SD | p-value
-------------------------------------|--------|--------|---------
(Intercept)                          |   2.1  |  0.3   | <1e-3
salience_labellow                    |  -0.7  |  0.3   | 0.021
winstart_labellate                   |   0.1  |  0.3   | 0.836
salience_labellow:winstart_labellate |   0.2  |  0.4   | 0.694
(phi)                                |   3.8  |  0.7   | <1e-3

Coefficient                     | Df  | Sum Sq | Mean Sq | F value | Pr(>F)
--------------------------------|-----|--------|---------|---------|---------
salience_label                  | 1   | 0.1275 | 0.12747 |  5.310  | 0.0238 *
winstart_label                  | 1   | 0.0001 | 0.00012 |  0.005  | 0.9430
salience_label:winstart_label   | 1   | 0.0008 | 0.00084 |  0.035  | 0.8518
Residuals                       | 80  | 1.9204 | 0.02401 |

### Spline regression

Coefficient                               | mean  | sd   | 10%  | 50%  | 90%
------------------------------------------|-------|------|------|------|-----
(Intercept)                               |  0.5  | 0.1  |  0.4 |  0.5 |  0.6
s(winstart):salience_labelhigh.1          |  0.0  | 0.3  | -0.3 |  0.0 |  0.4
s(winstart):salience_labelhigh.2          |  0.0  | 0.3  | -0.4 |  0.0 |  0.3
s(winstart):salience_labelhigh.3          |  0.0  | 0.3  | -0.4 |  0.0 |  0.3
s(winstart):salience_labelhigh.4          | -0.1  | 0.3  | -0.5 | -0.1 |  0.2
s(winstart):salience_labelhigh.5          |  0.0  | 0.3  | -0.3 |  0.0 |  0.3
s(winstart):salience_labelhigh.6          |  0.0  | 0.2  | -0.2 |  0.0 |  0.2
s(winstart):salience_labelhigh.7          | -0.3  | 0.2  | -0.5 | -0.3 |  0.0
s(winstart):salience_labelhigh.8          |  0.0  | 0.2  | -0.3 |  0.0 |  0.3
s(winstart):salience_labelhigh.9          | -0.2  | 0.4  | -0.7 | -0.1 |  0.3
s(winstart):salience_labellow.1           |  0.1  | 1.3  | -1.4 |  0.1 |  1.8
s(winstart):salience_labellow.2           |  0.4  | 1.1  | -1.0 |  0.4 |  1.7
s(winstart):salience_labellow.3           |  2.9  | 1.6  |  1.0 |  2.7 |  5.1
s(winstart):salience_labellow.4           |  0.8  | 0.8  | -0.2 |  0.7 |  1.8
s(winstart):salience_labellow.5           |  0.5  | 0.8  | -0.5 |  0.5 |  1.5
s(winstart):salience_labellow.6           | -1.6  | 0.4  | -2.1 | -1.6 | -1.1
s(winstart):salience_labellow.7           | -0.7  | 0.3  | -1.1 | -0.6 | -0.4
s(winstart):salience_labellow.8           | -1.3  | 0.7  | -2.2 | -1.3 | -0.5
s(winstart):salience_labellow.9           |  0.8  | 1.3  | -0.3 |  0.4 |  2.6
(phi)                                     |  2.7  | 0.0  |  2.6 |  2.7 |  2.7
smooth_sd[s(winstart):salience_labelhigh1]|  0.3  | 0.2  |  0.1 |  0.3 |  0.6
smooth_sd[s(winstart):salience_labelhigh2]|  0.7  | 0.7  |  0.1 |  0.5 |  1.6
smooth_sd[s(winstart):salience_labellow1] |  1.6  | 0.7  |  0.9 |  1.5 |  2.5
smooth_sd[s(winstart):salience_labellow2] |  1.1  | 1.0  |  0.1 |  0.8 |  2.5

## Target time

Coefficient                   |  Df | Sum Sq | Mean Sq | F value | Pr(>F)
------------------------------|-----|--------|---------|---------|---------
target_time_label             |   1 | 0.0342 | 0.03423 |  2.549  | 0.1140
condition                     |   1 | 0.0802 | 0.08023 |  5.974  | 0.0165 *
target_time_label:condition   |   1 | 0.0003 | 0.00032 |  0.024  | 0.8777
Residuals                     |  88 | 1.1818 | 0.01343 |

Coefficient                            | mean |  sd | 10% |  50% |  90% | p-value
---------------------------------------|------|-----|-----|------|------|-------
(Intercept)                            |  0.1 | 0.1 | 0.0 |  0.1 |  0.3 | 0.248
target_time_labellate                  |  0.3 | 0.2 | 0.1 |  0.3 |  0.5 | 0.066
conditionspatial                       |  0.4 | 0.2 | 0.2 |  0.4 |  0.6 | 0.020
target_time_labellate:conditionspatial |  0.7 | 0.3 | 0.4 |  0.7 |  1.0 | 0.004
(phi)                                  | 11.8 | 1.8 | 9.6 | 11.7 | 14.2 | 0.000

## Target time x Salience

Coefficient                                |   Df  | Sum Sq| Mean Sq | F value| Pr(>F)
-------------------------------------------|-------|-------|---------|--------|---------
target_time_label                          |   1   | 0.082 | 0.08245 |  3.226 | 0.0742 .
condition                                  |   1   | 0.144 | 0.14378 |  5.625 | 0.0188 *
salience_label                             |   1   | 0.018 | 0.01784 |  0.698 | 0.4046
target_time_label:condition                |   1   | 0.001 | 0.00076 |  0.030 | 0.8636
target_time_label:salience_label           |   1   | 0.088 | 0.08812 |  3.447 | 0.0650 .
condition:salience_label                   |   1   | 0.009 | 0.00887 |  0.347 | 0.5565
target_time_label:condition:salience_label |   1   | 0.002 | 0.00207 |  0.081 | 0.7764
Residuals                                  |   176 | 4.499 | 0.02556 |        |

Coefficient                                              |Median| MAD_SD| p-value
---------------------------------------------------------|------|-------|-------
(Intercept)                                              |  0.6 |  0.2  | <1e-3
salience_labellow                                        | -0.8 |  0.2  | <1e-3
target_time_labellate                                    | -0.6 |  0.2  | 0.009
conditionspatial                                         |  0.4 |  0.2  | 0.069
salience_labellow:target_time_labellate                  |  1.8 |  0.3  | <1e-3
salience_labellow:conditionspatial                       | -0.1 |  0.3  | 0.749
target_time_labellate:conditionspatial                   |  0.9 |  0.3  | 0.003
salience_labellow:target_time_labellate:conditionspatial | -0.5 |  0.4  | 0.253
(phi)                                                    |  6.2 |  0.7  | <1e-3