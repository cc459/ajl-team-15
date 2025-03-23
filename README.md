# Equitable AI for Dermatology (AJL Team 15) 

---

### **👥 Team Members**

| Name | GitHub Username | Contribution |
| ----- | ----- | ----- |
| Cheyanne Atole | @ | Research, data preprocessing, model building |
| Claire Cao | @cc459 | Research, data preprocessing, model building |
| Bijju Marharjan | @bijyeta-maharjan | Research, data preprocessing, model building |
| Kathy Yang | @katherinehyang | Research, data preprocessing, model building |


---

## **🎯 Project Highlights**

**Example:**

* Convolutional Neural Network (CNN) using transfer learning (EfficientNet-B0) to classify dermatological images by skin condition
* Achieved an F1 score of 0.20064 and a ranking of 48 on the final Kaggle Leaderboard
* Used \[explainability tool\] to interpret model decisions
* Implemented image augmentation and random sampling to optimize results within compute constraints

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

1. **Clone the repository**

    ```bash
    git clone https://github.com/[INSERT_REPO_URL]
    cd [REPO_NAME]
    ```

2. **Install dependencies**

    We recommend using a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the environment**

    Ensure you have Python 3.8+ and necessary libraries (e.g., PyTorch, torchvision, scikit-learn, pandas, matplotlib, albumentations).

4. **Access the dataset(s)**

    - Download the dataset from the Kaggle competition page.
    - Place the dataset files in the `bttai-ajl-2025/` directory (create this folder if it doesn’t exist).

5. **Run the notebook**

    Open and run the main Jupyter notebook (e.g., `AJL_Team_15.ipynb`). Note that our current setup is designed to work within a shared Google drive. You may need to adjust specific setup commands or directory pathways accordingly.

---

## **🏗️ Project Overview**

**Describe:**

* The Kaggle competition and its connection to the Break Through Tech AI Program
* The objective of the challenge
* The real-world significance of the problem and the potential impact of your work

This project is part of the **Break Through Tech AI Program**, which connects students with real-world machine learning problems.

We participated in the **Equitable AI for Dermatology** Kaggle competition, hosted in collaboration with the **Algorithmic Justice League (AJL)**, to address bias and fairness in dermatological AI systems.

The objective was to build a classification model that performs equitably across different skin tones. The real-world impact of this work lies in ensuring diagnostic tools do not perpetuate healthcare disparities, particularly for underrepresented groups.

---

## **📊 Data Exploration**

We used the provided **dermatological image dataset** from Kaggle, which included metadata like diagnosis and skin tone.

**Data Exploration & Preprocessing:**
- Analyzed class distribution and skin tone representation
- Applied data augmentation (horizontal flips, random crops, brightness adjustment)
- Resized all images to 224x224
- Addressed class imbalance via weighted sampling and augmentation

**Challenges & Assumptions:**
- Limited samples for certain skin tones
- Variability in image quality and lighting
- Assumed accurate skin tone labels from the dataset metadata

**Potential visualizations to include:**
* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## **🧠 Model Development**

**Model:** EfficientNet-B0 pretrained on ImageNet, fine-tuned for classification  

**Approach:**
- Used transfer learning to reduce training time and resource needs
- Freeze initial layers, fine-tuned final layers
- Used stratified split for train/val/test sets (80/10/10)
- Evaluation metric: Macro F1-score

**Hyperparameter Tuning:**
- Learning rate search via grid search
- Epochs: 
- Batch size: 

**(As applicable):**
* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

---

## **📈 Results & Key Findings**

**Performance Metrics:**
- F1 Score: 0.20064
- Kaggle Ranking: 48

**Fairness Evaluation:**

**Visualizations:**

**(As applicable):**

* Performance metrics (e.g., Kaggle Leaderboard score, F1-score)
* How your model performed overall
* How your model performed across different skin tones (AJL)
* Insights from evaluating model fairness (AJL)

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## **🖼️ Impact Narrative**

**Answer the relevant questions below based on your competition:**

**AJL challenge:**

As Dr. Randi mentioned in her challenge overview, “Through poetry, art, and storytelling, you can reach others who might not know enough to understand what’s happening with the machine learning model or data visualizations, but might still be heavily impacted by this kind of work.”
As you answer the questions below, consider using not only text, but also illustrations, annotated visualizations, poetry, or other creative techniques to make your work accessible to a wider audience.
Check out [this guide](https://drive.google.com/file/d/1kYKaVNR\_l7Abx2kebs3AdDi6TlPviC3q/view) from the Algorithmic Justice League for inspiration!

1. What steps did you take to address [model fairness](https://haas.berkeley.edu/wp-content/uploads/What-is-fairness_-EGAL2.pdf)? (e.g., leveraging data augmentation techniques to account for training dataset imbalances; using a validation set to assess model performance across different skin tones)

- Applied data augmentation to boost samples for underrepresented skin tones  
- Assessed validation performance across skin tone groups using disaggregated metrics  

2. What broader impact could your work have?

Our work contributes to raising awareness about the inequities embedded in medical datasets and how AI models can inadvertently propagate these biases. By highlighting disparities in performance, we can advocate for better data collection, representation, and auditing practices in AI for healthcare.

We envision a future where dermatological diagnostic tools are **built equitably from the ground up**, empowering clinicians to serve diverse communities more accurately.

---


---

## **🚀 Next Steps & Future Improvements**

**Address the following:**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

**Limitations:**
- Limited dataset size for darker skin tones as well as accurately classified images affected model fairness  
- CNNs sometimes focused on irrelevant background features  

**Future Work:**
- Fine-tune on external datasets with better diversity  
- Integrate metadata (e.g., age, gender) into multimodal models  
- Use techniques like re-weighted loss functions or adversarial training for fairness  

---

## **📄 References & Additional Resources**

- **Starter Code**  
  [AJL Starter Notebook on Kaggle](https://www.kaggle.com/code/cindydeng424/ajl-starter-notebook/notebook)

- **Image Processing for Machine Learning**  
  [Google Colab: Image Processing](https://colab.research.google.com/drive/1-ItNcRMbZBE6BCwPT-wD8m3YmHqwHxme?usp=sharing)

- **Data Sampling for Computer Vision**  
  [Google Colab: Data Sampling](https://colab.research.google.com/drive/1BD_qA5ptemVWbQSgw24t61H4kbwJOJxX?usp=sharing)

- **Data Augmentation for Computer Vision**  
  [Google Colab: Data Augmentation](https://colab.research.google.com/drive/1_-uxtcakO814BjLkqX6Z2I8Z7kqLs3Vh?usp=sharing)

- **Transfer Learning**  
  [Google Colab: Transfer Learning](https://colab.research.google.com/drive/14LCAi7usxBlVW1dwLXl7vAV0aRmzwEE9?usp=sharing)

- **Algorithmic Fairness in Practice**  
  [Google Colab: Algorithmic Fairness](https://colab.research.google.com/drive/1l5qSIbH-Gxj-ctTzVkfLLx6UU8mXJ45L?usp=share_link)
---
