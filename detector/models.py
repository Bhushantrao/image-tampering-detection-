from django.db import models


class UploadedImage(models.Model):

    image = models.ImageField(
        upload_to='uploaded_images/'
    )

    ela_image = models.ImageField(
        upload_to='ela_images/',
        blank=True,
        null=True
    )

    # New Fields
    localized_image = models.ImageField(
        upload_to='localized_images/',
        blank=True,
        null=True
    )

    heatmap_image = models.ImageField(
        upload_to='heatmaps/',
        blank=True,
        null=True
    )

    explainable_image = models.ImageField(
        upload_to='explainable/',
        blank=True,
        null=True
    )

    prediction = models.CharField(
        max_length=20,
        blank=True
    )

    confidence = models.FloatField(
        default=0
    )

    tampered_percentage = models.FloatField(
        default=0
    )

    uploaded_at = models.DateTimeField(
        auto_now_add=True
    )

    tampered_regions_json = models.TextField(
        blank=True,
        default='[]'
    )

    def __str__(self):
        return f"{self.id} - {self.prediction}"