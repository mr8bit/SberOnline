from .models import ReviewModel

from rest_framework import serializers


class ReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReviewModel
        fields = '__all__'
