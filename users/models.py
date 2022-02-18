from django.db import models

class UserRegistrationModel(models.Model):
    idno=models.AutoField(primary_key=True)
    user_name=models.CharField(max_length=100,unique=True)
    current_address=models.CharField(max_length=100)
    email=models.EmailField(unique=True)
    password=models.CharField(max_length=100)
    status=models.CharField(max_length=50,default=True)
