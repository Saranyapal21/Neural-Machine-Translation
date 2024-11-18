# Neural-Machine-Translation
Pre-trained models are trained on extremely large corpuses. Thus, these models are able to capture more patterns from the data and thus perform better compared to the model built from scratch. Our model which is built from scratch could not be trained on extremely large corpuses due to the availability of limited resources with us.  

In this part of the project, we have used a pre-trained model, specifically, `facebook/m2m100_418M`, which is a multilingual machine translation model from Facebook's M2M100 series. For our use case, we required a pre-trained model which is not very big, but still performs well on most sentences. Bigger models lead to slower computations, thus requiring more time for the model to load. This hampers user experience. Thus, we opt for a smaller model which still performs reasonably on our inputs.

The `facebook/m2m100_418M` model is built on the Transformer model architecture. It is designed for multilingual translation without reliance on English as a pivot language. It is suitable mostly for tasks involving language translation across diverse languages, including low-resource languages.
