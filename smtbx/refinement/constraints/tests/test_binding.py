"""Quick binding test."""
from smtbx.refinement import constraints as _
print("Import OK")
print("nomore_u_star_parameter:", hasattr(_, 'nomore_u_star_parameter'))

# Try to instantiate
from scitbx.array_family import flex
print("Trying to get constructor signature...")
try:
  print(_.nomore_u_star_parameter.__init__.__doc__)
except Exception as e:
  print("Error:", e)
