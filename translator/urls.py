from django.urls import path
from translator import views

urlpatterns = [
    path("", views.translateRequest, name="translateRequest"),
    path("translate", views.translateResponse, name="translateResponse"),
    path("completeSentence", views.completeSentence, name="completeSentence"),
    path("nextSentence", views.nextSentence, name="nextSentence"),
]