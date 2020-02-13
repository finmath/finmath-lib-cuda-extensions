package net.finmath.util;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;

import net.finmath.montecarlo.opencl.RandomVariableOpenCL;

public class FileUtils {

	/**
	 * Get the input stream of a resource file.
	 * 
	 * @param resourceName File name of the resource (absolute path).
	 * @return Corresponding input stream.
	 * @throws URISyntaxException Thrown if the URI was malformed.
	 * @throws IOException Thrown if the file could not be opened.
	 */
	public static InputStream getInputStreamForResource(String resourceName) throws URISyntaxException, IOException {
		// Get the input stream of a resource (may be in a Jar file).
		final URI uri = RandomVariableOpenCL.class.getClassLoader().getResource(resourceName).toURI();
		return getInputStreamForURI(uri);	
	}

	/**
	 * Get the input stream of a resource file.
	 * 
	 * @param resourceName File name of the resource (absolute path).
	 * @return Corresponding input stream.
	 * @throws URISyntaxException Thrown if the URI was malformed.
	 * @throws IOException Thrown if the file could not be opened.
	 */
	public static InputStream getInputStreamForURI(final URI uri) throws URISyntaxException, IOException {
		// Get the input stream of an uri (may be in a Jar file).
		try
		{
			FileSystems.getFileSystem(uri);
		}
		catch( FileSystemNotFoundException e )
		{
			Map<String, String> env = new HashMap<>();
			env.put("create", "true");
			FileSystems.newFileSystem(uri, env);
		}
		return Files.newInputStream(Paths.get(uri));

	}

	/**
	 * Fully reads the given InputStream and returns it as a string.
	 *
	 * @param inputStream The input stream to read.
	 * @return The contents of the input stream as string.
	 */
	public static String readToString(final InputStream inputStream)
	{
		try(BufferedReader br = new BufferedReader(new InputStreamReader(inputStream)))
		{
			final StringBuffer sb = new StringBuffer();
			String line = null;
			while (true)
			{
				line = br.readLine();
				if (line == null)
				{
					break;
				}
				sb.append(line).append("\n");
			}
			return sb.toString();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
			System.exit(1);
			return null;
		}
	}

	/**
	 * Fully reads the given InputStream and returns it as a byte array.
	 *
	 * @param inputStream The input stream to read
	 * @return The byte array containing the data from the input stream
	 * @throws IOException If an I/O error occurs
	 */
	public static byte[] readToByteArray(final InputStream inputStream) throws IOException
	{
		final ByteArrayOutputStream baos = new ByteArrayOutputStream();
		final byte buffer[] = new byte[8192];
		while (true)
		{
			final int read = inputStream.read(buffer);
			if (read == -1)
			{
				break;
			}
			baos.write(buffer, 0, read);
		}
		return baos.toByteArray();
	}

	public static void writeInputStreamToFile(final InputStream inputStream, final File targetFile) throws IOException {
		java.nio.file.Files.copy(
				inputStream, 
				targetFile.toPath(), 
				StandardCopyOption.REPLACE_EXISTING);
	}
}
