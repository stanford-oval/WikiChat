// ...library and helper functions for upload page
function updateFileName() {
    const fileInput = document.getElementById('file');
    const fileNameDisplay = document.getElementById('file-name');
    const fileName = fileInput.files.length > 0 ? fileInput.files[0].name : 'No file chosen';
    fileNameDisplay.textContent = fileName;
}

// Function to smoothly scroll to the expanded section
function scrollToExpandedSection(collapseElement) {
    if (collapseElement.checked) {
        collapseElement.closest('.collapse').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('file').addEventListener('change', updateFileName);
    const collapses = document.querySelectorAll('.collapse input[type="checkbox"]');
    collapses.forEach(collapse => {
        collapse.addEventListener('change', function () {
            scrollToExpandedSection(collapse);
        });
    });
});
