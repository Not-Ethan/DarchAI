async function saveEvidence(data) {
    const response = await fetch('/save-evidence', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({data})
    });
  
    // Check if the response has a file
    if (response.headers.get('Content-Type') === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'evidence.docx';
      link.click();
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
      }, 0);
    } else {
      // Handle any errors
      console.error('Failed to download the file');
    }
  }
  document.addEventListener("DOMContentLoaded", function () {
    const evidenceUrls = document.querySelectorAll(".evidence-url");

    evidenceUrls.forEach((url) => {
      url.addEventListener("click", (event) => {
        event.stopPropagation();
        window.open(url.href, "_blank");
      });
    });
  });
  document.getElementById("download-selected-evidence").addEventListener("click", function() {
const checkboxes = document.querySelectorAll(".evidence-checkbox");
const selectedEvidence = [];

checkboxes.forEach(checkbox => {
  if (checkbox.checked) {
    selectedEvidence.push(checkbox.dataset.evidence);
  }
});

saveEvidence(selectedEvidence);
});
  