Time-resolved scRNA-seq
=======================

RNA velocity
------------

Although the seminal RNA velocity work is exciting, it has the following
limitations:

1. It can only predict short-term direction and magnitude of RNA
   dynamics.
2. It is mostly a descriptive instead of a predictive tool.
3. It relies on the ``mis-priming`` of intron reads for current
   single-cell platforms and thus the intron measures are biased and
   inaccurate.
4. RNA velocity was estimated as :math:`U - \gamma / \beta S`
   (:math:`U`: unspliced RNA, :math:`S`: spliced RNA, :math:`\gamma`:
   degradation rate, :math:`\beta`: splcing rate, :math:`\gamma / \beta`
   is the slope of the steady state cell fitting.), it is thus scaled by
   the splicing rate and lacks real physical meanings (i.e. molecules /
   hour).

We reason that metabolic labeling based method which measures both the
historical or old, and the new and nascent RNA of cells in a
controllable way will be better measurements for RNA velocity and
transcriptomic dynamics. When extending metabolic labeling to single
cell RNA-seq, labeling based scRNA-seq essentially measures two
modalities or timepoints for the same cell.

How does metabolic labeling work
--------------------------------

How can we quantify nascent RNA via metabolic labeling? Overall there
are two different methods, the biotin purification or chemical
conversion based approach. Both approaches are quiet similar in that we
first need to applying different labeling strategies to label the cells.
For biotin purification, we need to use thiol-specific biotinylation to
tag labeled mRNA. Then the streptavidin beads can be used to pull down
and separate the pre-exisiting RNA and newly transcribed RNA. Then we
will follow by preparing two separate libraries, old and new RNAs, for
sequencing. There are a few very well known issue regarding this method:

1. it often introduces 20-30% cross-contanimation between old and new
   RNAs,
2. it also leads to some normalization issues between different
   libraries.

On the other hand, the chemical conversion based approaches avoid the
labrious and error-prone procedure of separating old/old RNA and
preparing two different libraries and emerged as the favored strategy
recently. The key idea of chemical conversion based methods are that by
some chemical reaction we can artificially introduce T to C mutation
which can then be used to distinuigh labelled and thus new RNA from old
RNA. There are about three different chemistry developed: IAA alkylation
or hydrogen bond reconfiguration via TimeLapse-seq or TUC-seq chemistry.

In fact, metabolic labeling has been widely adapted for the past a few
decades. We can use various nucleotides to label RNA, for example, BrU,
Eu and Biotin-NTP. For 4sU based labeling, there are about three
different strategies, namely, SLAM-seq, TUC-seq, and Time-lapse-seq.

Metabolic labeling based scRNA-seq
----------------------------------

Recently a few groups adapted the bulk method to either the plate-based
scRNA-seq with SMART-seq2 method, for example,
`scSLAM-seq <https://www.nature.com/articles/s41586-019-1369-y>`__ or
`NASC-seq <https://www.nature.com/articles/s41467-019-11028-9>`__.
`scEU-seq <https://science.sciencemag.org/content/367/6482/1151.full>`__
is based on
`CEL-Seq2 <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0938-8>`__
and is also plate-based but uses UMI in contrast to scSLAM-seq or
NASC-seq. The scEU-seq method is based on EU and required purification
and it thus may involve cross-contanimation or normalization issues.

`Cao, et
al <https://www.nature.com/articles/s41587-020-0480-9#:~:text=Abstract,not%20directly%20capture%20transcriptional%20dynamics.&text=We%20used%20sci%2Dfate%20to,in%20%3E6%2C000%20single%20cultured%20cells>`__
recently developed sci-fate which integrates 4sU labeling and
combinatorial indexing based scRNA-seq so it can potentially enables
measuring hundread thousands of single cells.

For the first time, `Wu lab <https://www.wulabupenn.org/>`__ from Upenn
developed a drop-seq based metabolic labeling based scRNA-seq, scNT-seq.

Comparison between different labeling based scRNA-seq methods
-------------------------------------------------------------

In `Qiu, Hu, et.
al <https://www.nature.com/articles/s41592-020-0935-4>`__, we performed
a detailed comparison (Supplementary table 7) between scNT-seq with
other available methods. Especially for the improved second-strand
synthesis based strategy, we are able to obtain substantially high
number of genes and UMIs per cell with relatively few number of reads.
Thus scNT-seq is arguably one of the best metabolic labeling based
scRNA-seq strategies.

In our study, we show that dynamo can be used to leverage scNT-seq
datasets for time-resolved RNA-velocity analysis. Those results
demonstrate the power of dynamo and scNT-seq in revealing the
fine-grained transcriptomic dynamics.

Labeling strategies
-------------------

We can be very creative and smart in designing the metabolic labeling
experiments. For example, you can design an experiment where you can
take different days and perform a kinetic experiment at each day. This
can help you obtain transcription rate, splicing and degradation rate
over time. But this is often time-consuming, so we may just choose a
typical day for a single kinetic experiment. In addition, we may also
perform a degradation experiment where we label the cells with 4sU for
an extended time period to saturate the 4sU labeling in cells. Then we
can wash out the 4sU and replaced with excess U, followed by chasing at
different time points. This can help us to estimate the splicing and
degradation rates (and half life) of RNA. We can also just design a
one-shot labeling experiment to label cells at different time points.
Since splicing and degradation rate of mRNA is often constant, thus
combining one-shot experiments with degradation experiments, we are able
to get even more accurate estimates of the transcription rate at each
time point. We also want to note that we can combine different labeling
strategies, for exmple, combining pulse chase in a single experiment or
integrating metabolic labeling with drug treatment or genetic
perturbations.

Dynamo’s comprehensive model framework for analyzing lableing datasets
----------------------------------------------------------------------

In order to fully take advantage of the scSLAM-seq data, we recently
developed a sophisticated framework, dynamo that provides an inclusive
model of expression dynamics with scSLAM-seq and multiomics, vector
field reconstruction and potential landscape mapping. In dynamo, we
abstract every steps from RNA transcription, splicing, metabolic
labeling, translation and RNA or protein degradation. We can model the
mean and variance of RNA species via a set of moment equations, we then
transform them into a matrix format and solve them efficiently. In
dynamo, we also implemented the traditional RNA velocity method based on
the steady state assumptions to support analyzing regular 10 x data.
Similarly, dynamo supports studying cite-seq data to estimate protein
velocity.
