## Frequently Asked Questions
Before you post a new question, please first look at the following Q & A and existing GitHub issues.

#### If the ill-posed forward operator is known, seems to synthesize massive training pairs + supervised learning is enough to learn a reconstruction function? ([#3](https://github.com/edongdongchen/REI/issues/3))
Similar questions include “why learning image from raw measurements (or unsupervised learning) is useful and important?”.

(1) Having the ill-posed forward operator known is not saying we have the true signal model to the groundtruth raw images. (2) We can not synthesize the nontrivial data pairs if we don't know the signal model of interest. (3) We would like to emphasize that the distribution of synthetic data P_syn and that of the raw measurements P_raw are different. `Training with the synthetic data pairs can only build a data-driven prior to the synthetic signal model, it is not learning the true signal model of raw signals, i.e. the function learned on synthetic data can not reconstruct structures or patterns that are not present in the synthetic training pairs` (examples can be found in this issue [#3](https://github.com/edongdongchen/REI/issues/3)). In contrast, if unsupervised learning (or 'learn to image from raw measurements') is possible, such unseen/unpredictable structures/patterns that are embedded in the raw measurements can be therefore figured out.

#### What priors are Equivariant Imaging really exploited? is it includes inductive bias from network architecture (e.g. locality in CNN)?

The EI only takes the `group invariance` of the signal model. This prior doesn't include any inductive bias from network architectures. Both EI and the group invariance of the signal model are agnostic to network architectures.

#### Group invariance is a mild prior, why EI can learn the signal model? 

Fundamentally, the interaction between group action and nullspace can make the ill-posed problem to be well-posed. Under the sampling conditions, the model identification can be guaranteed in EI. Please check our [theorey paper](https://arxiv.org/pdf/2203.12513.pdf) for more details
