# OTTR

Supplementary code and data for our manuscript "*Organotypic Timelapse recording with Transcriptomic Readout (OTTR) links cell behaviour to cell identity in human tissues*" (Vinsland, Mannens et al., manuscript submitted).

<p>&nbsp;</p>

<img src="./static/ottr_overview.png" alt="OTTR overview" width="800px"/>

**Figure: OTTR. a)** Overview of OTTR's workflow; and **b)** data modalities
acquired.


## Preprint (bioRxiv)

https://www.biorxiv.org/content/10.1101/2025.07.28.667119v1

## Detailed protocols

https://www.protocols.io/view/ottr-gyydbxxs7

## Browser

Visualizations of the dataset will be browsable at CELLxGENE.

## Data availability

#### Raw sequence reads

- scRNA-seq BAM files will be available from the European Genome/Phenome Archive (https://ega-archive.org/) under accession number TBA. 

#### scRNA-seq expression matrices

- Complete count matrices (gene ✕ cell counts) for the cultured cortices are available as [.loom](https://loompy.org) files:

  [cultures_OneMIV_HS.loom](https://storage.googleapis.com/linnarsson-lab-ottr/cultures_OneMIV_HS.loom) (64 MB) - Cortical cultures maintained for One Month In Vitro (Fig.2, ExtD. Fig.2)

  [cultures_TwoWIV_HS.loom](https://storage.googleapis.com/linnarsson-lab-ottr/cultures_TwoWIV_HS.loom) (182 MB) - Cortical cultures maintained for Two Weeks In Vitro (Fig.2, ExtD. Fig.2)

  [cultures_EYFP.loom](https://storage.googleapis.com/linnarsson-lab-ottr/cultures_EYFP.loom) (387 MB) - Cortical cultures with lentiviral EYFP expression, cultured for 4 and 11 days in vitro (Fig.2, ExtD. Fig.2)
  
- The datasets can also be downloaded as .h5ad files from the browser: [CELLxGENE]. 

#### Xenium spatial data

- Raw Xenium data and images have been deposited at the [BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/) under accession number TBA.
- Raw Xenium data and images can also be downloaded directly from here:

  [output-XETG00045__0011077__EV37_TL__20231025__135741.gzip](https://storage.googleapis.com/linnarsson-lab-ottr/output-XETG00045__0011077__EV37_TL__20231025__135741.gzip) (4 GB)

  [output-XETG00045__0011077__EV39_con__20231025__135741.gzip](https://storage.googleapis.com/linnarsson-lab-ottr/output-XETG00045__0011077__EV39_con__20231025__135741.gzip) (4 GB)

  [output-XETG00045__0011080__EV39_TL_A__20231025__135741.gzip](https://storage.googleapis.com/linnarsson-lab-ottr/output-XETG00045__0011080__EV39_TL_A__20231025__135741.gzip) (4 GB)

  [output-XETG00045__0011080__EV39_TL_B__20231025__135741.gzip](https://storage.googleapis.com/linnarsson-lab-ottr/output-XETG00045__0011080__EV39_TL_B__20231025__135741.gzip) (3 GB)

  [output-XETG00045__0011080__EV39_TL_C__20231025__135741.gzip](https://storage.googleapis.com/linnarsson-lab-ottr/output-XETG00045__0011080__EV39_TL_C__20231025__135741.gzip) (4 GB)

  [output-XETG00047__0011072__EV37_con__20231025__140913.gzip](https://storage.googleapis.com/linnarsson-lab-ottr/output-XETG00047__0011072__EV37_con__20231025__140913.gzip) (4 GB)

  [output-XETG00047__0011075__EV38_TL_A__20231025__140913.gzip](https://storage.googleapis.com/linnarsson-lab-ottr/output-XETG00047__0011075__EV38_TL_A__20231025__140913.gzip) (6 GB)

  [output-XETG00047__0011075__EV38_TL_B__20231025__140913.gzip](https://storage.googleapis.com/linnarsson-lab-ottr/output-XETG00047__0011075__EV38_TL_B__20231025__140913.gzip) (32 GB)
  
After downloading, Xenium folders can be browsed using the free interactive [Xenium Explorer](https://www.10xgenomics.com/products/xenium-analysis) tool.

#### Timelapse videos

- Supplementary videos:

  [Video S1](https://storage.googleapis.com/linnarsson-lab-ottr/Video_S1_Cell_divisions.mp4): Cell divisions
  
  [Video S2](https://storage.googleapis.com/linnarsson-lab-ottr/Video_S2_cell_accumulation_around_vessels.mp4): Cell accumulation around vessels
  
  [Video S3](https://storage.googleapis.com/linnarsson-lab-ottr/Video_S3_vessel_migration.mp4): Vessel migration
  
  [Video S4](https://storage.googleapis.com/linnarsson-lab-ottr/Video_S4_Vessel_zoom_cell_behaviour.mp4): Vessel zoom cell behavior

- Raw timelapse images have been deposited at the [BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/) under accession number TBA.
- Cell tracks can be downloaded from here:

  TBD

## Code used for analysis and visualisation

#### scRNA-seq

- Clustering of the scRNA-seq data was performed using the [cytograph-dev](https://github.com/linnarsson-lab/cytograph-dev) version of cytograph. This is the version used for our adult human brain project. Its installation and usage are described [here](https://github.com/linnarsson-lab/adult-human-brain/tree/main/cytograph). 
- [Jupyter notebooks](https://jupyter.org/) used to make figures. The notebooks require [cytograph-dev](https://github.com/linnarsson-lab/cytograph-dev). (NOTE: *cytograph-shoji* will *not* work).

#### Xenium and alignment with timelapse

- Jupyter notebooks to generate the figures can be found here [here](notebooks)
- Jupyter notebooks used for Xenium data processing are found [here].
- Notebooks for alignment: 
