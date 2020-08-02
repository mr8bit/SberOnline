from rest_framework import viewsets
from .serializers import ReviewSerializer
from .models import ReviewModel


class ReviewViewSet(viewsets.ModelViewSet):
    serializer_class = ReviewSerializer
    queryset = ReviewModel.objects.all()
