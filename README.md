# Real Fake Face Image Classifier "DeepFakes"

This repository contains the project notebook that was developed during the Module #3 *Deep learning, computer vision and evolutionary computation* course by TDX | University of Coimbra [Data Science Academy](https://www.talkdesk.com/tdx/data-science-academy) (Powered by Talkdesk). Total course time was 80 hours plus 3 additional weeks for project related work.

For the project work and with the provided dataset, a real-fake face classifier was developed using several approaches: *convolutional neural networks* (CNN) and transfer learning with and without face cropping using the *Multi-Task Cascaded Convolutional Neural Networks* (MTCNN).

To extend the course work, and in order to leverage the created models, a web application was developed using the [streamlit](https://streamlit.io/) framework. For more information about the models or the project, check out the ```project``` folder in the root directory of this repository.

## View the web app live
Click [here](https://realfakefaceclassification.herokuapp.com/) to go to the app page.

## Run the app locally
```
git clone https://github.com/dvpinho/RealFakeFaces.git
cd RealFakeFaces/
pip install -r requirements.txt
streamlit run app.py
```
---

![app_screenshot](https://github.com/dvpinho/RealFakeFaces/blob/main/screenshots/app_screenshot.png)
