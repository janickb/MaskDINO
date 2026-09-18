The idea is that I train the model on generic or common medical instruments.


In a second step the backbone as well as the transformer encoder / decoder should be frozen and only the classifier linear layer should be retrained based on 10-20 images from instruments with a new class. I want to see if the feature embeedings of the querries are diverse enough to classify these new or exchanged models also with just few-shot training.

- i want some sort of mode which should only be for retraining the classifier -> is this enough? what are you thoughts
- every instrument / class which should be detected will be supported with 10-20 images per class (different perspective etc.)
- at the end of retraining the classifier a short test should be done to see if the new classes are distinct of each other

"grill-me"!
