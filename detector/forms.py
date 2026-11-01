from django import forms

class UploadForm(forms.Form):

    image = forms.ImageField(
        widget=forms.ClearableFileInput(
            attrs={
                'class':'form-control'
            }
        )
    )