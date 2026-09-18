import org.gradle.api.tasks.compile.JavaCompile
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()

rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory =
        newBuildDir.dir(project.name)

    project.layout.buildDirectory.value(newSubprojectBuildDir)
}

subprojects {
    project.evaluationDependsOn(":app")
}

// tflite_flutter 0.12.1 sets compileOptions to VERSION_11 (Java 11).
// The Kotlin plugin for plugin subprojects defaults to JVM 17, causing the mismatch.
// We cannot override Java upward (afterEvaluate fails because :app is pre-evaluated
// by evaluationDependsOn above). Instead, we align Kotlin DOWN to JVM_11 for all
// plugin subprojects — matching their declared Java target.
// The :app project is excluded: it sets Java=17 + Kotlin=17 in its own build.gradle.kts.
subprojects {
    if (name != "app") {
        tasks.withType<KotlinJvmCompile>().configureEach {
            compilerOptions.jvmTarget.set(
                org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11
            )
        }
    }
}






tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}