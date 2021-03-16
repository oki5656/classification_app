import streamlit as st
import cv2

st.title("test showing an image")
img=cv2.imread("../../data/cat.jpg",1)
img=cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
cv2.image(img)