from django.shortcuts import render, get_object_or_404
from django.core.files.storage import FileSystemStorage
from .models import UploadedImage
from django.core.files import File
import os
import json
from django.http import JsonResponse
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta
from .predict import predict_and_explain


def home(request):

    if request.method == "POST":

        # Check image exists
        if 'image' not in request.FILES:

            return render(
                request,
                'upload.html',
                {
                    'error': 'Please select an image'
                }
            )

        file = request.FILES['image']

        # Allow only image files
        allowed_extensions = [
            '.jpg',
            '.jpeg',
            '.png'
        ]

        if not any(
            file.name.lower().endswith(ext)
            for ext in allowed_extensions
        ):

            return render(
                request,
                'upload.html',
                {
                    'error': 'Only JPG, JPEG and PNG files are allowed'
                }
            )

        try:

            fs = FileSystemStorage()

            filename = fs.save(
                file.name,
                file
            )

            image_url = fs.url(
                filename
            )

            full_path = fs.path(
                filename
            )

            # Call the unified prediction and explainability pipeline
            prediction, confidence, tampered_percentage, ela_path, localized_path, heatmap_path, explainable_path, regions = predict_and_explain(
                full_path
            )

            # -------------------------
            # Save Analysis to Database
            # -------------------------

            record = UploadedImage()

            # Original uploaded image
            record.image.name = filename

            # Prediction details
            record.prediction = prediction
            record.confidence = confidence
            record.tampered_percentage = tampered_percentage
            record.tampered_regions_json = json.dumps(regions)

            # Save ELA image
            if ela_path and os.path.exists(ela_path):
                with open(ela_path, "rb") as f:
                    record.ela_image.save(
                        f"ela_{filename}",
                        File(f),
                        save=False
                    )

            # Save Localized Image
            if localized_path and os.path.exists(localized_path):
                with open(localized_path, "rb") as f:
                    record.localized_image.save(
                        f"localized_{filename}",
                        File(f),
                        save=False
                    )

            # Save Heatmap
            if heatmap_path and os.path.exists(heatmap_path):
                with open(heatmap_path, "rb") as f:
                    record.heatmap_image.save(
                        f"heatmap_{filename}",
                        File(f),
                        save=False
                    )

            # Save Explainable Map
            if explainable_path and os.path.exists(explainable_path):
                with open(explainable_path, "rb") as f:
                    record.explainable_image.save(
                        f"explainable_{filename}",
                        File(f),
                        save=False
                    )

            record.save()

            # Clean up temp files
            for p in [ela_path, localized_path, heatmap_path, explainable_path]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

            if prediction == 'Fake':
                explanation = f'''
                Image classified as Fake.

                Suspicious regions detected using Vision Transformer (ViT) self-attention.

                Tampered Area: {tampered_percentage}%
                '''
            else:
                explanation = '''
                Image classified as Original.

                No significant manipulation detected by the Vision Transformer model.
                '''

            return render(
                request,
                'result.html',
                {
                    'prediction': prediction,
                    'confidence': confidence,
                    'original_image': image_url,
                    'ela_image': record.ela_image.url if record.ela_image else '',
                    'localized_image': record.localized_image.url if record.localized_image else '',
                    'heatmap_image': record.heatmap_image.url if record.heatmap_image else '',
                    'explainable_image': record.explainable_image.url if record.explainable_image else '',
                    'tampered_percentage': tampered_percentage,
                    'explanation': explanation,
                    'regions': regions
                }
            )

        except Exception as e:

            return render(
                request,
                'upload.html',
                {
                    'error': f'Error: {str(e)}'
                }
            )

    return render(
        request,
        'upload.html'
    )
    
def history(request):

    records = UploadedImage.objects.order_by("-uploaded_at")

    return render(
        request,
        "history.html",
        {
            "history": records
        }
    )

def result_detail(request, pk):
    record = get_object_or_404(UploadedImage, pk=pk)
    
    try:
        regions = json.loads(record.tampered_regions_json) if record.tampered_regions_json else []
    except Exception:
        regions = []
        
    if record.prediction == 'Fake':
        explanation = f'''
        Image classified as Fake.

        Suspicious regions detected using Vision Transformer (ViT) self-attention.

        Tampered Area: {record.tampered_percentage}%
        '''
    else:
        explanation = '''
        Image classified as Original.

        No significant manipulation detected by the Vision Transformer model.
        '''
        
    return render(
        request,
        'result.html',
        {
            'prediction': record.prediction,
            'confidence': record.confidence,
            'original_image': record.image.url,
            'ela_image': record.ela_image.url if record.ela_image else '',
            'localized_image': record.localized_image.url if record.localized_image else '',
            'heatmap_image': record.heatmap_image.url if record.heatmap_image else '',
            'explainable_image': record.explainable_image.url if record.explainable_image else '',
            'tampered_percentage': record.tampered_percentage,
            'explanation': explanation,
            'regions': regions
        }
    )
    
def analytics(request):
    return render(request, "analytics.html")

def analytics_data(request):
    # 1. Total records
    total = UploadedImage.objects.count()
    if total == 0:
        return JsonResponse({
            'total': 0,
            'prediction_distribution': {'labels': [], 'data': []},
            'avg_confidence': 0.0,
            'avg_tampered_pct': 0.0,
            'timeline': {'labels': [], 'data': []}
        })
        
    # 2. Prediction distribution
    dist = UploadedImage.objects.values('prediction').annotate(count=Count('id'))
    fake_count = 0
    original_count = 0
    for d in dist:
        pred_val = str(d['prediction']).strip().lower()
        if 'fake' in pred_val or 'tamp' in pred_val:
            fake_count += d['count']
        else:
            # Everything else (Real, Original, empty, etc.) is grouped under Original
            original_count += d['count']
            
    pred_labels = []
    pred_data = []
    if fake_count > 0:
        pred_labels.append('Fake Image')
        pred_data.append(fake_count)
    if original_count > 0:
        pred_labels.append('Original Image')
        pred_data.append(original_count)
        
    # 3. Average metrics
    avg_conf = UploadedImage.objects.aggregate(avg=Avg('confidence'))['avg'] or 0.0
    avg_tamp = UploadedImage.objects.aggregate(avg=Avg('tampered_percentage'))['avg'] or 0.0
    
    # 4. Timeline (last 7 days)
    today = timezone.now().date()
    timeline_labels = []
    timeline_data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = UploadedImage.objects.filter(uploaded_at__date=day).count()
        timeline_labels.append(day.strftime("%d %b"))
        timeline_data.append(count)
        
    return JsonResponse({
        'total': total,
        'prediction_distribution': {
            'labels': pred_labels,
            'data': pred_data
        },
        'avg_confidence': round(avg_conf, 2),
        'avg_tampered_pct': round(avg_tamp, 2),
        'timeline': {
            'labels': timeline_labels,
            'data': timeline_data
        }
    })

