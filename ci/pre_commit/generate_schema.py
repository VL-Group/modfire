import sys
from marshmallow_jsonschema import JSONSchema
import importlib.util
import json
import copy

modfire_config = importlib.util.spec_from_file_location("modfire.config", sys.argv[1])
config = importlib.util.module_from_spec(modfire_config)
modfire_config.loader.exec_module(config)

configSchema = config.ConfigSchema()

json_schema = JSONSchema()
result = json_schema.dump(configSchema)



with open(sys.argv[2], "w") as fp:
    json.dump(result, fp)

with open(sys.argv[2]) as fp:
    lines = fp.readlines()

with open(sys.argv[2], "w") as fp:
    for line in lines:
        fp.write(line.replace("/components/schemas", ""))

print(f"Generate {sys.argv[2]} done.")



from json_schema_for_humans.generation_configuration import GenerationConfiguration
from json_schema_for_humans.generate import generate_from_filename


config_badge = GenerationConfiguration(
    template_name="md",
    template_md_options={"badge_as_image": True},
    deprecated_from_description=True,
    footer_show_time=False, show_breadcrumbs=False, examples_as_yaml=True
)

generate_from_filename(sys.argv[2], sys.argv[3], copy_css=False, copy_js=False, config=config_badge)
