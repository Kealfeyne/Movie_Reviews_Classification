from django.http.response import HttpResponse
from django.shortcuts import get_object_or_404, render
from reviews.models import ReviewForm
from ml_model.predict import predict


def index(request):
    submitbutton = request.POST.get("submit")

    content = ''

    form = ReviewForm(request.POST or None)
    if form.is_valid():
        content = form.cleaned_data.get("content")

    predict_id, binary_predict_id = predict(content)

    context = {'form': form, 'content': content, 'submitbutton': submitbutton, 'predict_id': predict_id,
               'binary_predict_id': binary_predict_id}

    return render(request, 'index.html', context)
