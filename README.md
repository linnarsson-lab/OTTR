# OTTR

Supplementary code and data to our study of OTTR: Organotypic Timelapse recording with Transcriptomic Readout (OTTR) links cell
identity to cell behaviour in human tissues. Vinsland, Mannens et al. (manuscript in preparation).

![ottr_overview](./static/ottr_overview.png)
**Figure: OTTR. a)** Overview of OTTR's workflow; and **b)** data modalities
acquired.


## Preprint (bioRxiv)

TBA

## Browser

Visualizations of the dataset are browsable at CELLxGENE.

## Data availability

#### Raw sequence reads

- scRNA-seq BAM files will be available from the European Genome/Phenome Archive (https://ega-archive.org/) under accession number TBA. 

#### scRNA-seq expression matrices

- Complete count matrices (gene x cell counts) for the cultured cortices are available as loom files [here].
- The datasets can also be downloaded as .h5ad files from the browser: [CELLxGENE]. 

#### Xenium spatial data

- Raw Xenium data and images have been deposited at the [BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/) under accession number TBA.
- Complete count matrices (gene x cell counts) are available as loom files [here].

## Code used for analysis and visualisation

- Clustering of the scRNA-seq data was performed using the cytograph-dev version of cytograph. This is the version used for our adult human brain project. Its installation and usage are described [here](https://github.com/linnarsson-lab/adult-human-brain/tree/main/cytograph). 
- [Jupyter](https://jupyter.org/) notebooks used to make figures are available [here]. The notebooks also import from cytograph-dev. (cytograph-shoji will *not* work).
- Jupyter notebooks used for Xenium data processing are found [here].
