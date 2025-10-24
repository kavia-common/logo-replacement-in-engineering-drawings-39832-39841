# Frontend PDF Support

- Update the drawings ZIP content helper text to say it accepts images and PDFs:
  "Upload a ZIP containing PNG, JPG/JPEG, TIFF, BMP, GIF, or PDF drawings."

- Ensure your file input for the logo remains image-only, but the drawings input should allow `.zip`:
  `<input type="file" accept=".zip" />`

- No change is needed to the request body; backend now handles PDFs inside the ZIP.
