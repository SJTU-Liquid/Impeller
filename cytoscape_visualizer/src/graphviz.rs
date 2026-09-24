use super::DigraphBuilder;
use std::io;

impl DigraphBuilder {
    pub fn emit_graphviz(&self, f: &mut impl io::Write) -> io::Result<()> {
        // Write the DOT header
        writeln!(f, "digraph ComputationGraph {{")?;
        writeln!(f, "  // Graph settings")?;
        writeln!(f, "  graph [")?;
        writeln!(f, "    fontname = \"Helvetica\"")?;
        writeln!(f, "    fontsize = 11")?;
        writeln!(f, "  ]")?;

        // Node settings
        writeln!(f, "  // Node settings")?;
        writeln!(f, "  node [")?;
        writeln!(f, "    fontname = \"Helvetica\"")?;
        writeln!(f, "    fontsize = 11")?;
        writeln!(f, "    shape = \"box\"")?;
        writeln!(f, "    style = \"rounded\"")?;
        writeln!(f, "  ]")?;

        // Edge settings
        writeln!(f, "  // Edge settings")?;
        writeln!(f, "  edge [")?;
        writeln!(f, "    fontname = \"Helvetica\"")?;
        writeln!(f, "    fontsize = 11")?;
        writeln!(f, "  ]")?;

        Ok(())
    }
}
