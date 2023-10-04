from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from . import views
from .views import generate_pdf

urlpatterns = [
    path("", views.index, name="index"),
path('generate_pdf/', generate_pdf, name='generate_pdf'),
]

if settings.DEBUG:
     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)