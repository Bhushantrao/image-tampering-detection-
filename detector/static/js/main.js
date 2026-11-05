console.log("ELA-Enhanced Vision Transformer for Image Manipulation Detection System Loaded");

// =======================
// Image Preview
// =======================

document.addEventListener("DOMContentLoaded", function () {

    const input = document.getElementById("imageInput");

    const preview = document.getElementById("preview");

    if (input) {

        input.addEventListener("change", function () {

            const file = this.files[0];

            if (file) {

                preview.src = URL.createObjectURL(file);

                preview.style.display = "block";

            }

        });

    }

});

// =======================
// Loading Animation
// =======================

document.addEventListener("DOMContentLoaded", function () {

    const form = document.getElementById("uploadForm");

    if (form) {

        form.addEventListener("submit", function () {

            document.getElementById("loading").style.display = "flex";

        });

    }

});