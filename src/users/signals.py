from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import User
from rest_framework.authtoken.models import Token


@receiver(post_save, sender=User)
def make_token(sender, instance, created, **kwargs):
    if created:
        Token.objects.create(user=instance)
