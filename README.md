# Missing Migrants

## Summary about the Missing Migrants Project
The Missing Migrants Project is an initiative aimed at collecting and analyzing data on migrants who have died or gone missing during the process of migration. Managed by the International Organization for Migration (IOM), the project brings together a variety of sources, including media reports, government records, and interviews, to provide a comprehensive look at the risks and challenges faced by migrants worldwide. The data is intended to inform policy, raise awareness, and serve as a resource for further research. https://missingmigrants.iom.int/


# Missing Migrants Project Analysis

This repository contains a series of Jupyter notebooks that analyze data from the Missing Migrants Project. The notebooks explore various aspects of the data, from data cleaning and exploration to in-depth analysis of specific migration routes.

## Notebooks Overview

### missing_migrants_datasets_comparison.ipynb

**Objective:** Compare multiple datasets to ensure data integrity and consistency.

**Key Highlights:**
- Data loading and preliminary exploration.
- Comparison of records across datasets.
- Visualization of missing records.

### missing_migrants_eda.ipynb

**Objective:** Conduct an exploratory data analysis (EDA) on the primary dataset.

**Key Highlights:**
- Data cleaning and preprocessing.
- Visualization of key metrics and distributions.
- Geospatial exploration of incident locations.

### missing_migrants_migration_route_round_[1-3].ipynb

**Objective:** Predict missing migration routes using machine learning models.

**Key Highlights:**
- Data preprocessing specific to model training.
- Model selection and evaluation.
- Feature importance and model insights.

### missing_migrants_hypothesis_testing.ipynb

**Objective:** Test hypotheses related to deadliest migration routes and times.

**Key Highlights:**
- One-way ANOVA test for migration routes.
- Tukey's Honest Significant Difference (HSD) test for pairwise comparison of routes.
- Seasonal analysis of incidents.

### missing_migrants_central_mediterranean.ipynb

**Objective:** Detailed analysis of the Central Mediterranean migration route.

**Key Highlights:**
- Geospatial visualization of incident locations.
- Exploration of deadliest segments along the route.
- Identification of common causes of incidents.

### missing_migrants_route_selector.ipynb

**Objective:** Interactive exploration of different migration routes.

**Key Highlights:**
- Route selection using widgets.
- Geospatial visualizations tailored to selected routes.

### missing_migrants_sahara.ipynb

**Objective:** Comprehensive analysis of the Sahara Desert crossing route.

**Key Highlights:**
- Visualization of common countries of origin.
- Identification of deadliest segments along the route.
- Exploration of primary causes of incidents.

## Datasets

The primary dataset used across these notebooks originates from the Missing Migrants Project, which tracks incidents involving migrants, including refugees and asylum-seekers, who have died or gone missing in the process of migration towards an international destination.

## Visualizations

All the plots and visualizations generated during the analysis are stored in the `Pictures` folder. This includes:

- Geospatial visualizations showcasing incident locations.
- Bar charts, pie charts, and line graphs illustrating various metrics and distributions.
- Heatmaps and cluster plots highlighting hotspots and concentrations of incidents.

To view the visualizations:

1. Navigate to the `Pictures` folder in the repository.
2. Open the desired image file to view the visualization.

![Description of the Image](/pictures/map.png)

## Presentation Overview

For a summarized overview of the analysis and key findings, you can refer to the included presentation:

[Missing Migrants Project Analysis Presentation](./slides/MISSING_MIGRANTS_PROJECT.pdf)

This presentation provides a concise walkthrough of the project, highlighting the main insights and visualizations from the analysis.

## Interactive Visualizations on Tableau Public

For a more interactive exploration of the data and additional visualizations, check out our dashboard on Tableau Public:

[Missing Migrants Project Analysis Dashboard](https://public.tableau.com/app/profile/lilit.davidyan/viz/missing_migrants/WorldMap)

This dashboard provides dynamic visualizations, allowing you to dive deeper into specific aspects of the data and customize your view.


