from django.shortcuts import render,redirect
from django.views.generic import TemplateView, ListView
from django.views.generic import View
from django.contrib import messages

from users.models import UserRegistrationModel


class User_login(TemplateView):
    template_name = "user_login.html"
    extra_context ={"data":"data"}

class Admin_login(TemplateView):
    template_name = "admin_login.html"

class Admin_login_check(View):
    def post(self,requset):
        uname=requset.POST.get("uname")
        pword=requset.POST.get("pword")
        if uname=="admin" and pword=="admin":
            return render(requset,"admin_home.html")
        else:
            messages.success(requset,"Invalid Details")
            return redirect('admin_login')
class User_details_save(View):
    def post(self,request):
        uname=request.POST.get("uname")
        pword=request.POST.get("pword")
        email=request.POST.get("email")
        address=request.POST.get("address")
        status="pending"
        UserRegistrationModel.objects.create(user_name=uname,email=email,password=pword,current_address=address,status=status)
        messages.success(request,"User Registered Successfully")
        return redirect('user_register')
class User_requests(ListView):
    model = UserRegistrationModel
    queryset = UserRegistrationModel.objects.filter(status="pending")
    template_name = 'admin_home.html'
    context_object_name ="Udata"

class Approve_user(View):
    def get(self,request,id):
        qs=UserRegistrationModel.objects.filter(idno=id)
        qs.update(status="approved")
        return render(request,"admin_home.html",{"data":"User Approved"})

class Decline_user(View):
    def get(self,request,id):
        qs=UserRegistrationModel.objects.filter(idno=id)
        qs.update(status="declined")
        return render(request,"admin_home.html",{"data":"User Declined"})

class User_Login_Validate(View):
    def post(self,request):
        uname=request.POST.get("uname")
        pword=request.POST.get("pword")
        qs=UserRegistrationModel.objects.filter(user_name=uname,password=pword)
        for x in qs:
            status=x.status

        if qs and status=="approved":
            return render(request,"user_home.html",{"name":uname})
        elif qs and (status=="pending" or status=="declined"):
            messages.success(request, "Your details need to approve by admin..please wait until approve..!")
            return redirect('user_login')
        else:
            messages.success(request,"Invalid Details")
            return redirect('user_login')
class User_logout(View):
    def get(self,request):
        messages.success(request, "Sucessfully Logged Out")
        return redirect('main')

from users.Algorithm import logistic,forest,d_tree,predict

class Logistic(View):
    def get(self,req):
        algo=req.GET.get("algo")
        print(algo,"==================================")
        re=logistic()
        print(re,"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        accuracy=str((re[0]*100))
        return render(req,"user_home.html",{"algo_name":"Logistic Regression","accuracy":accuracy,"precision":str(re[1])})


class Forest(View):
    def get(self,req):
        algo=req.GET.get("algo")
        re=forest()
        accuracy=str((re[0]*100))
        return render(req,"user_home.html",{"algo_name":"Random Forest Classification","accuracy":accuracy,"precision":str(re[1])})


class Tree(View):
    def get(self, req):
        algo = req.GET.get("algo")
        re = d_tree()
        accuracy = str((re[0] * 100))
        return render(req, "user_home.html",{"algo_name": "Decision Tree Classification", "accuracy": accuracy, "precision": str(re[1])})


class Predict(View):
    def post(self,req):
        age=int(req.POST.get("age"))
        gender=int(req.POST.get("gender"))
        scholor=int(req.POST.get("scholor"))
        hyper=int(req.POST.get("hyper"))
        handi=int(req.POST.get("handi"))
        dia=int(req.POST.get("dia"))
        habbit=int(req.POST.get("habbit"))
        sms=int(req.POST.get("sms"))
        a=[age,gender,scholor,hyper,dia,handi,habbit,sms,1,0,0,0,0]
        pred=predict(a)
        if pred[0]==0:
            result="Showed up"
        else:
            result="Not Showed"
        return render(req,"pred.html",{"re":result})



class Admin_logout(View):
    def get(self,request):
        messages.success(request, "Sucessfully Logged Out")
        return redirect('main')


class User_info(ListView):
    model = UserRegistrationModel
    template_name = 'admin_home.html'
    context_object_name ="uidata"
