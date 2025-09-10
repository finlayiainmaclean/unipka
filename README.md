### Unofficial inference wrapper around DPTech's UnipKa model

This repo provides a simple wrapper around Zheng and co's fantastic [UnipKa](https://pubs.acs.org/doi/10.1021/jacsau.4c00271) model. It refactors their [example notebook](https://www.bohrium.com/notebooks/38543442597) into a small python package. 

This repo provides:

- A macro and micro pKa calculator, including pH-adjusted free energies for each microstate
- A logD calculator following the methodology of [Rowan Sci](https://chemrxiv.org/engage/chemrxiv/article-details/68388349c1cb1ecda02ba65d), by calculating the weighted average of logP values for each microstate. 
- A state penalty function, useful for calculating permeability of neutral species, also following the methodology of [Rowan Sci](https://chemrxiv.org/engage/chemrxiv/article-details/68388349c1cb1ecda02ba65d) and [Lawrenz and co](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00150). 
- A solvation energy calculator that uses xTB and the ALPB implicit solvent model.
- A Kpuu calculator built using Kpuu, logD and the state penalty as features.
- A small Jupyter widget to visualise the microstate distributions across a range of pH values.

Please cite the corresponding authors if you use this wrapper in your work!

![](unipka.gif)

Install using `pip install unipka`


Please see `examples.ipynb` or the unit tests for example usage.


Performance on the SAMPL benchmarks (pKa prediction):

<img src="benchmarks/sampl_results.png">

Performance on logD benchmark:

<img src="benchmarks/logd_results.png" width="600">


Performance on Kpuu benchmark:

<img src="benchmarks/kpuu_results.png" width="600">

Performance on FreeSolv benchmark (solvation prediction):

<img src="benchmarks/solvation_results.png" width="600">




