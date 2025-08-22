Rodar o train_target
`
python -m runner.train_target --model [mlp, cnn] --scenario [single, batch]
`

Rodar o invert
`
python -m runner.invert --model mlp --attack dlg --defense none --opt lbfgs --scenario single
`