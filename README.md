# Elucidating Transcriptomic Profiles from Single-cell RNA sequencing Data using Nature-Inspired Compressed Sensing
Gene expression profiling can define the cell state and gene-expression pattern of cells at the genetic level in a high-throughput manner. With the development of transcriptome techniques, processing high-dimensional genetic data has become a major challenge in expression profiling. Thanks to the recent widespread use of matrix decomposition methods in bioinformatics, a computational framework based on compressed sensing was adopted to reduce dimensionality. However, compressed sensing requires an optimization strategy to learn the modular dictionaries and activity levels from the low-dimensional random composite measurements for reconstructing the high-dimensional gene expression data. Considering this, we introduce and compare three compressed sensing frameworks coming from nature-inspired optimization algorithms (CSCS, ABCCS, and FACS) to improve the quality of the decompression process. Several experiments establish that the proposed methods outperform benchmark methods on nine different datasets, especially the FACS method. Therefore, we illustrate the robustness and convergence of FACS in various aspects. Notably, the time complexity and parameter analyse various properties of our proposed FACS. Furthermore, differential gene expression analysis, cell type clustering, gene ontology enrichment, and pathology analysis are conducted that bring novel insights into cell type identification and characterization mechanisms from different perspectives. All algorithms are written in Python and available at https://github.com/Philyzh8/Nature-inspired-CS

## Citation

```
@article{yu2021elucidating,
         title={Elucidating transcriptomic profiles from single-cell RNA sequencing data using nature-inspired compressed sensing},
         author={Yu, Zhuohan and Bian, Chuang and Liu, Genggeng and Zhang, Shixiong and Wong, Ka-Chun and Li, Xiangtao},
         journal={Briefings in Bioinformatics},
         volume={22},
         number={5},
         pages={bbab125},
         year={2021},
         publisher={Oxford University Press}
         }
```
