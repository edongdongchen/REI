## Frequently Asked Questions
Before you post a new question, please first look at the following Q & A and existing GitHub issues.

#### If the ill-posed forward operator is known, can't we create synthetic training pairs and used supervised learning to learn a reconstruction function? ([#3](https://github.com/edongdongchen/REI/issues/3))
Similar questions include “why learning image from raw measurements (or unsupervised learning) is useful and important?”.

Knowing the forward operator known does not provide any knowledge about the true signal model. In real-world settings, the distribution of synthetic data P_syn and that of the raw measurements P_raw are different. `Training with synthetic data will only work if the synthetic model matches the true signal model - the function learned on synthetic data cannot reconstruct structures or patterns that are not present in the synthetic training pairs` (examples can be found in this issue [#3](https://github.com/edongdongchen/REI/issues/3)). In contrast, unsupervised learning bypasses this problem, since test and train data are necessarily associated to the same underlying distribution.

#### What priors is Equivariant Imaging really exploiting? is it includes inductive bias from network architecture (e.g. locality in CNN)?

The EI only assumes `invariance to certain transformations (e.g., shifts, rotations, etc)` of the signal model. This prior is not related to any inductive bias associated with the network architectures.

#### Group invariance is a mild prior, why EI can learn the signal model? 

Fundamentally, the interaction between group action and nullspace can make the ill-posed problem to be well-posed. Under some mild conditions on the number of measurements per signal, model identification can be guaranteed - please check our [theorey paper](https://arxiv.org/pdf/2203.12513.pdf) for more details!
