if [ -e "pyproject.toml" ]; then
  echo "Found pyproject.toml"
else
  echo "Run cd .."
  cd ..
fi

echo "Remove The Old Build"
rm -rf dist/ build/ *.egg-info/

echo "Start Build..."
python -m build

whl_file=$(ls dist/*.whl 2>/dev/null)
base_name="${whl_file%.whl}"
IFS='-' read -ra parts <<< "$base_name"
num_parts=${#parts[@]}
version="${parts[$((num_parts - 4))]}"

echo "Extracted version: $version"

if [[ "$version" =~ a[0-9]+$ ]]; then
    echo "Version $version is an alpha pre-release. Uploading..."
    twine upload --repository Lingxi_Mlkit "$whl_file"
else
    echo "Version $version is NOT an alpha pre-release. Skipping upload."
    exit 1
fi