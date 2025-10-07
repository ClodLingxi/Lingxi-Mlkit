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

echo "Copy Dist File To ~/Download"
WHEEL_FILE=$(ls dist/lingxi_mlkit-*-py3-none-any.whl)
cp "$WHEEL_FILE" ~/Downloads/