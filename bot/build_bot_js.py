"""Embed padel-bot-weights.json into padel-bot.js."""
import json, os
d = os.path.dirname(os.path.abspath(__file__))
model = json.load(open(os.path.join(d, 'padel-bot-weights.json')))
tpl = open(os.path.join(d, 'padel-bot.template.js')).read()
open(os.path.join(d, 'padel-bot.js'), 'w').write(tpl.replace('__MODEL__', json.dumps(model, separators=(',', ':'))))
print('wrote padel-bot.js')
