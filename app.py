import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
from huggingface_hub import hf_hub_download

class MultiTaskDRModel(nn.Module):
    def __init__(self, model_name='efficientnet_b3', num_classes=5,
                 num_lesion_types=5, num_regions=5, pretrained=False):
        super(MultiTaskDRModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.feature_dim, self.feature_dim // 8), nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 8, self.feature_dim), nn.Sigmoid()
        )
        self.feature_norm = nn.BatchNorm1d(self.feature_dim)
        self.dropout = nn.Dropout(0.4)
        self.severity_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(self.feature_dim // 2, num_classes)
        )
        self.lesion_detector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(self.feature_dim // 4, num_lesion_types)
        )
        self.region_predictor = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(self.feature_dim // 4, num_regions)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x); pooled_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        attention_weights = self.attention(pooled_features.unsqueeze(-1).unsqueeze(-1)); features = pooled_features * attention_weights
        features = self.feature_norm(features); features = self.dropout(features)
        severity_logits = self.severity_classifier(features); lesion_logits = self.lesion_detector(features); region_logits = self.region_predictor(features)
        return {'severity': severity_logits, 'lesions': lesion_logits, 'regions': region_logits, 'features': features}

class MedicalKnowledgeBase:
    def __init__(self):
        self.severity_descriptions = {0: "No Diabetic Retinopathy", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
        self.recommendations = {
            0: "No signs of diabetic retinopathy detected. Continue with annual routine screening.",
            1: "Mild signs detected. An ophthalmology referral within 6-12 months is recommended for monitoring. Maintain good glycemic control.",
            2: "Moderate signs detected. An ophthalmology referral within 3-6 months is recommended. Closer monitoring and strict glycemic control are advised.",
            3: "Severe signs detected. An URGENT ophthalmology referral within 1-2 months is critical. Intensive diabetic management is required.",
            4: "Proliferative signs detected. An IMMEDIATE ophthalmology referral is required. This is a vision-threatening stage that may require urgent intervention (e.g., laser therapy or injections)."
        }
    def generate_description_template(self, severity_level, lesion_findings, region_findings, confidence=0.9):
        base_desc = self.severity_descriptions.get(severity_level, "Unknown Stage"); description = f"**Diagnosis:** {base_desc}\n\n**Confidence:** {confidence:.1%}\n\n"
        active_lesions = [lesion.replace('_', ' ').title() for lesion, present in lesion_findings.items() if present]
        if active_lesions: description += f"**Potential Clinical Findings:**\n- {', '.join(active_lesions)}\n\n"
        else:
            if severity_level > 0: description += "**Potential Clinical Findings:**\n- General signs of retinopathy corresponding to the diagnosed stage.\n\n"
        description += f"**Recommendation:**\n- {self.recommendations.get(severity_level, 'Consult an ophthalmologist.')}"
        return description

@st.cache_resource
def load_model_from_hf(repo_id, filename):
    """Downloads the model from Hugging Face Hub and loads it into memory."""
    with st.spinner(f"Downloading model '{filename}' from Hugging Face Hub... This may take a moment."):
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    with st.spinner("Loading model into memory..."):
        device = torch.device('cpu')
        model = MultiTaskDRModel()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    return model

def get_image_transforms():
    return A.Compose([A.Resize(512, 512), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB"); image_np = np.array(image)
    transforms = get_image_transforms(); transformed = transforms(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)
    return image_tensor, image

def get_prediction(model, image_tensor):
    with torch.no_grad(): outputs = model(image_tensor)
    severity_probs = torch.softmax(outputs['severity'], dim=1)
    confidence, predicted_class = torch.max(severity_probs, dim=1)
    lesion_probs = torch.sigmoid(outputs['lesions']).squeeze().numpy(); region_probs = torch.sigmoid(outputs['regions']).squeeze().numpy()
    lesion_types = ['microaneurysms', 'hemorrhages', 'hard_exudates', 'soft_exudates', 'neovascularization']
    region_types = ['superior', 'inferior', 'nasal', 'temporal', 'macular']
    lesion_findings = {name: prob > 0.5 for name, prob in zip(lesion_types, lesion_probs)}
    region_findings = {name: prob > 0.5 for name, prob in zip(region_types, region_probs)}
    return {"severity_level": predicted_class.item(), "confidence": confidence.item(), "lesion_findings": lesion_findings, "region_findings": region_findings}

st.set_page_config(page_title="Diabetic Retinopathy Diagnosis AI", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("About the Project")
st.sidebar.info("This web application is a demonstration of a deep learning pipeline for the automated grading of Diabetic Retinopathy (DR). It uses a state-of-the-art EfficientNet-B3 model hosted on Hugging Face Hub.")
st.sidebar.title("Model Performance (V2)")
st.sidebar.markdown("- **Quadratic Weighted Kappa (QWK):** 0.796\n- **Accuracy:** 65.0%\n- **F1-Score (Weighted):** 66.3%\n\n*This V2 model was specifically optimized to improve the QWK score, a more clinically relevant metric for DR grading.*")
st.sidebar.warning("**Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice.")

st.title("👁️ Automated Diabetic Retinopathy Diagnosis")
st.markdown("Upload a retinal fundus image to receive an automated diagnostic assessment based on the ICDRS severity scale.")

# --- 3. HUGGING FACE CONFIGURATION ---
HF_REPO_ID = "dheeren-tejani/DiabeticRetinpathyClassifier" 
HF_FILENAME = "best_model_v2.pth"

try:
    model = load_model_from_hf(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    st.sidebar.success("Model loaded successfully from Hugging Face Hub!")
except Exception as e:
    st.error(f"Error loading model from Hugging Face Hub: {e}")
    st.info("Please ensure the HF_REPO_ID is correct and the model file exists.")
    st.stop() # Stop the app if the model can't be loaded

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    medical_kb = MedicalKnowledgeBase()
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Uploaded Image")
        st.image(uploaded_file, caption="Patient's Retinal Fundus Image", use_container_width=True)

    with st.spinner("The AI is analyzing the image..."):
        image_bytes = uploaded_file.getvalue()
        image_tensor, _ = preprocess_image(image_bytes)
        prediction = get_prediction(model, image_tensor)
        report = medical_kb.generate_description_template(prediction["severity_level"], prediction["lesion_findings"], prediction["region_findings"], prediction["confidence"])

    with col2:
        st.header("Diagnostic Report")
        severity_level = prediction["severity_level"]
        if severity_level >= 3: st.error(report)
        elif severity_level >= 1: st.warning(report)
        else: st.success(report)
        with st.expander("Show Detailed Model Outputs"):
            st.write("**Raw Severity Prediction:**"); st.write(f"Class {prediction['severity_level']} ({medical_kb.severity_descriptions[prediction['severity_level']]})")
            st.write("**Lesion Presence Predictions (Confidence > 50%):**"); st.json({k: v for k, v in prediction['lesion_findings'].items()})
            st.write("**Affected Region Predictions (Confidence > 50%):**"); st.json({k: v for k, v in prediction['region_findings'].items()})
else:
    st.info("Please upload an image file to begin analysis.")