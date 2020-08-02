# Generated by Django 3.0.8 on 2020-08-01 19:00

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ReviewModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('datetime_create', models.DateTimeField(auto_created=True, auto_now_add=True)),
                ('title', models.CharField(max_length=150, verbose_name='Название')),
                ('review', models.TextField(verbose_name='Отзыв')),
                ('platform', models.CharField(choices=[('a', 'Android'), ('i', 'iOS')], max_length=3, verbose_name='Платформа')),
            ],
        ),
    ]