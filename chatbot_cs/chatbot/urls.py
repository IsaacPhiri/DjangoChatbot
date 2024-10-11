from django.urls import path
from .views import chatbot_response, index

urlpatterns = [
    path('', index, name='index'),
    path('get_response/', chatbot_response, name='chatbot_response'),
]
