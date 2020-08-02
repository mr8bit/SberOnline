from django.db import models

# Create your models here.


class ReviewModel(models.Model):
    title = models.CharField(max_length=150, verbose_name='Название')
    review = models.TextField(verbose_name="Отзыв")
    platform = models.CharField(choices=(('a','Android'),('i', 'iOS')), max_length=3, verbose_name="Платформа")
    datetime_create = models.DateTimeField(auto_now_add=True, auto_created=True)