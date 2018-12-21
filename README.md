Siemese Network 
===

### General Strategy
---

1) Train a model to discriminate
between a collection of same/different pairs.
2) Generalize to
evaluate new categories based on learned feature mappings for
verification.

#### Filters and Activations
---
- filters of varying
size and a fixed stride of 1
- ReLU activation function
to the output feature maps
-  followed by maxpooling
with a filter size and stride of 2.

#### Loss function
---
![Loss](/Data/loss.jpeg)

#### Overall Model was as Follow
---
![Model](/Data/model.png)
