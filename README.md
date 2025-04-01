# Equitable AI for Dermatology (AJL Team 15) 

---

### **👥 Team Members**

| Name | GitHub Username | Contribution |
| ----- | ----- | ----- |
| Cheyanne Atole | @ | Exploratory research, data preprocessing, model building, model finetuning |
| Claire Cao | @cc459 | Exploratory research, data preprocessing, model building, model finetuning |
| Bijju Marharjan | @bijyeta-maharjan | Exploratory research, data preprocessing, model building, model finetuning |
| Kathy Yang | @katherinehyang | Exploratory research, data preprocessing, model building, model finetuning |


---

## **🎯 Project Highlights**

* Convolutional Neural Network (CNN) using transfer learning (ResNet50) to classify dermatological images by skin condition
* Achieved an F1 score of 0.20064 and a ranking of 48 on the final Kaggle Leaderboard
* Implemented image augmentation and random sampling to optimize results within compute constraints

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

1. **Clone the repository**

    ```bash
    git clone https://github.com/cc459/ajl-team-15
    cd ajl-team-15
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

    Open and run the main Jupyter notebook (`AJL_Team_15.ipynb`). Note that our current setup is designed to work within a shared Google drive. You may need to adjust specific setup commands or directory pathways accordingly.

---

## **🏗️ Project Overview**

This project is part of the **Break Through Tech AI Program**, which connects students with real-world machine learning problems.

We participated in the **Equitable AI for Dermatology** Kaggle competition, hosted in collaboration with the **Algorithmic Justice League (AJL)**, to address bias and fairness in dermatological AI systems.

The objective was to build a classification model that performs equitably across different skin tones. The real-world impact of this work lies in ensuring diagnostic tools do not perpetuate healthcare disparities, particularly for underrepresented groups.

---

## **📊 Data Exploration**

We used the provided **dermatological image dataset** from Kaggle, which included metadata like diagnosis and skin tone.

**Data Exploration & Preprocessing:**
- Analyzed class distribution and skin tone representation
- Applied data augmentation (horizontal flips, random crops, brightness adjustment)
- Resized all images to a standard shape
- Addressed class imbalance via weighted sampling and augmentation
- Filtered data by quality (e.g., diagonistic, characteristic)

**Challenges & Assumptions:**
- Limited samples for certain skin tones
- Variability in image quality and lighting
- Assumed accurate skin tone labels from the dataset metadata

---

## **🧠 Model Development**

**Model:** ResNet50 pretrained on ImageNet, fine-tuned for classification

**Approach:**
- Used transfer learning to reduce training time and resource needs
- Added custom dense and dropout layers on top of the ResNet base
- Trained using early stopping and model checkpoint callbacks
- Filtered training data by diagnostic quality for better consistency
- Evaluation metric: Macro F1-score

**Hyperparameter Tuning:**
- Optimizer: Adam with learning rate tuning
- Epochs: ~15–30 depending on early stopping
- Batch size: ~32

---

## **📈 Results & Key Findings**

**Performance Metrics:**
- F1 Score: 0.20064
- Kaggle Ranking: 48

**Fairness Evaluation:**
- Evaluated performance across disaggregated skin tone groups
- Observed discrepancies in prediction accuracy across tones

---

## **🖼️ Impact Narrative**

As Dr. Randi mentioned in her challenge overview, “Through poetry, art, and storytelling, you can reach others who might not know enough to understand what’s happening with the machine learning model or data visualizations, but might still be heavily impacted by this kind of work.”

**Model Fairness:**

We applied data augmentation to increase samples for underrepresented skin tones. We also assessed validation performance across skin tone groups using disaggregated metrics. In addition, we filtered high-quality images to reduce noise and potential bias.

**Broader Impact:**

Our work contributes to raising awareness about the inequities embedded in medical datasets and how AI models can inadvertently propagate these biases. By highlighting disparities in performance, we can advocate for better data collection, representation, and auditing practices in AI for healthcare.

We envision a future where dermatological diagnostic tools are **built equitably from the ground up**, empowering clinicians to serve diverse communities more accurately.

---

## **🚀 Next Steps & Future Improvements**

**Limitations:**
- Limited dataset size for darker skin tones as well as accurately classified images affected model fairness  
- CNNs sometimes focused on irrelevant background features  

**Future Work:**
- Fine-tune on external datasets with better diversity  
- Integrate more metadata into multimodal models  
- Use techniques such as adversarial training to improve fairness 
  
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
We would like to acknowledge and thank our amazing TA Ishita for her help throughout this process!