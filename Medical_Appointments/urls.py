"""Medical_Appointments URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from Medical_Appointments import settings
from users import views
from django.views.generic import TemplateView
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', TemplateView.as_view(template_name="index.html"), name='main'),
    path('user_login/', views.User_login.as_view(), name='user_login'),
    path('admin_login/', views.Admin_login.as_view(), name='admin_login'),
    path('admin_login_check/', views.Admin_login_check.as_view(), name='admin_login_check'),
    path('user_register/', TemplateView.as_view(template_name="user_register.html"), name='user_register'),
    path('user_details_save/', views.User_details_save.as_view(), name='user_details_save'),
    path('user_requests/', views.User_requests.as_view(), name='user_requests'),
    path('approve_user<int:id>/', views.Approve_user.as_view(), name='approve_user'),
    path('decline_user<int:id>/', views.Decline_user.as_view(), name='decline_user'),
    path('login/', views.User_Login_Validate.as_view()),
    path('user_logout/', views.User_logout.as_view(), name='user_logout'),
    path('logistic/', views.Logistic.as_view(), name='logistic'),
    path('forest/', views.Forest.as_view(), name='forest'),
    path('tree/', views.Tree.as_view(), name='tree'),
    path('predict/', views.Predict.as_view(), name='predict'),
    path('show/', TemplateView.as_view(template_name="pred.html")),
    path('admin_logout/', views.Admin_logout.as_view(), name='admin_logout'),
    path('user_info/', views.User_info.as_view(), name='user_info'),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)