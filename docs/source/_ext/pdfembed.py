# We would like to acknowledge the contribution of SuperKogito for their valuable code of sphinxcontrib-pdfembed.
# The original code can be found at https://github.com/SuperKogito/sphinxcontrib-pdfembed/blob/master/sphinxcontrib/pdfembed.py.

from docutils import nodes

def pdfembed_html(pdfembed_specs):
    """
    Build the iframe code for the pdf file,
    """
    html_base_code = """
                        <iframe
                                id="ID"
                                style="border:1px solid #666CCC"
                                title="PDF"
                                src="%s"
                                frameborder="1"
                                scrolling="auto"
                                height="%s"
                                width="%s"
                                align="%s">
                        </iframe>
                     """
    return ( html_base_code % (pdfembed_specs['src'   ],
                               pdfembed_specs['height'],
                               pdfembed_specs['width' ],
                               pdfembed_specs['align' ]) )

def pdfembed_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Get iframe specifications and generate the associate HTML code for the pdf iframe.
    """
    # parse and init variables
    text           = text.replace(' ', '')
    pdfembed_specs = {}
    # read specs
    for component in text.split(','):
         pdfembed_specs[component.split(':')[0]] = component.split(':')[1]
    # build node from pdf iframe html code
    node = nodes.raw('', pdfembed_html(pdfembed_specs), format='html')
    return [node], []

def setup(app):
    """
    Set up the app with the extension function
    """
    app.add_role('pdfembed', pdfembed_role)