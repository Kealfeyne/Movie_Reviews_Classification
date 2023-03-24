from django import forms
from django.db import models


# Create your models here.


class ReviewForm(forms.Form):
    content = forms.CharField(label="Review text", widget=forms.Textarea)
