import streamlit as st
import gdown
import joblib  # أو pickle اعتمادًا على كيفية حفظ النموذج

# رابط التحميل المباشر للنموذج من Google Drive
url = 'https://drive.google.com/file/d/1tm26hgFqH6jgquktn3ZosbTuRV_Yoepq/view?usp=sharing'

# مسار حفظ النموذج على الجهاز
output = 'model.pkl'

# تحميل النموذج
gdown.download(url, output, quiet=False)

# تحميل النموذج باستخدام joblib أو pickle
model = joblib.load(output)
# بناء واجهة المستخدم
st.title('Diabetes Prediction')

# إدخال البيانات من المستخدم
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=140, value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=0.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input('Age', min_value=0, max_value=120, value=0)

# زر للتنبؤ
if st.button('Predict'):
    # جمع البيانات المدخلة في قائمة
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]

    # استخدام النموذج للتنبؤ
    prediction = model.predict(input_data)

    # عرض النتيجة
    if prediction[0] == 1:
        st.write('The model predicts that the person has diabetes.')
    else:
        st.write('The model predicts that the person does not have diabetes.')

